# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import threading
import uuid
from abc import abstractmethod, ABCMeta
from typing import Sequence, Any, Dict, Callable, Mapping, Optional

import s3fs
import xarray as xr
import zarr

from xcube.constants import FORMAT_NAME_LEVELS
from xcube.constants import FORMAT_NAME_NETCDF4
from xcube.constants import FORMAT_NAME_SCRIPT
from xcube.constants import FORMAT_NAME_ZARR
from xcube.core.dsio import guess_dataset_format
from xcube.core.dsio import is_s3_url
from xcube.core.dsio import parse_s3_fs_and_root
from xcube.core.dsio import write_cube
from xcube.core.gridmapping import GridMapping
from xcube.core.normalize import decode_cube
from xcube.core.schema import rechunk_cube
from xcube.core.verify import assert_cube
from xcube.util.assertions import assert_instance
from xcube.util.perf import measure_time
from xcube.util.tilegrid import TileGrid

COMPUTE_DATASET = 'compute_dataset'


class MultiLevelDataset(metaclass=ABCMeta):
    """
    A multi-level dataset of decreasing spatial resolutions (a multi-resolution pyramid).

    The pyramid level at index zero provides the original spatial dimensions.
    The size of the spatial dimensions in subsequent levels
    is computed by the formula ``size[index + 1] = (size[index] + 1) // 2``
    with ``size[index]`` being the maximum size of the spatial dimensions at level zero.

    Any dataset chunks are assumed to be the same in all levels. Usually, the number of chunks is one
    in one of the spatial dimensions of the highest level.
    """

    @property
    @abstractmethod
    def ds_id(self) -> str:
        """
        :return: the dataset identifier.
        """

    @property
    @abstractmethod
    def grid_mapping(self) -> GridMapping:
        """
        :return: the CF-conformal grid mapping
        """

    @property
    @abstractmethod
    def tile_grid(self) -> TileGrid:
        """
        :return: the tile grid.
        """

    @property
    def num_levels(self) -> int:
        """
        :return: the number of pyramid levels.
        """
        return self.tile_grid.max_level + 1

    @property
    def base_dataset(self) -> xr.Dataset:
        """
        :return: the base dataset for lowest level at index 0.
        """
        return self.get_dataset(0)

    @property
    def datasets(self) -> Sequence[xr.Dataset]:
        """
        Get datasets for all levels.

        Calling this method will trigger any lazy dataset instantiation.

        :return: the datasets for all levels.
        """
        return [self.get_dataset(index) for index in range(self.num_levels)]

    @abstractmethod
    def get_dataset(self, index: int) -> xr.Dataset:
        """
        :param index: the level index
        :return: the dataset for the level at *index*.
        """

    def close(self):
        """ Close all datasets. Default implementation does nothing. """

    def apply(self,
              function: Callable[[xr.Dataset, Dict[str, Any]], xr.Dataset],
              kwargs: Dict[str, Any] = None,
              tile_grid: TileGrid = None,
              ds_id: str = None) -> 'MultiLevelDataset':
        """ Apply function to all level datasets and return a new multi-level dataset."""
        return MappedMultiLevelDataset(self, function, tile_grid=tile_grid, ds_id=ds_id, mapper_params=kwargs)


class LazyMultiLevelDataset(MultiLevelDataset, metaclass=ABCMeta):
    """
    A multi-level dataset where each level dataset is lazily retrieved, i.e. read or computed by the abstract method
    ``get_dataset_lazily(index, **kwargs)``.

    If no *tile_grid* is passed it will be retrieved lazily by using the ``get_tile_grid_lazily()`` method,
    which may be overridden.The default implementation computes a new tile grid based on the dataset at level zero.

    :param tile_grid: The tile grid. If None, a new tile grid will be computed based on the dataset at level zero.
    :param ds_id: Optional dataset identifier.
    :param parameters: Optional keyword arguments that will be passed to the ``get_dataset_lazily`` method.
    """

    def __init__(self,
                 grid_mapping: GridMapping = None,
                 tile_grid: TileGrid = None,
                 ds_id: str = None,
                 parameters: Mapping[str, Any] = None):
        self._grid_mapping = grid_mapping
        self._tile_grid = tile_grid
        self._ds_id = ds_id
        self._level_datasets: Dict[int, xr.Dataset] = {}
        self._parameters = parameters or {}
        self._lock = threading.RLock()

    @property
    def ds_id(self) -> str:
        if self._ds_id is None:
            with self._lock:
                self._ds_id = str(uuid.uuid4())
        return self._ds_id

    @property
    def grid_mapping(self) -> GridMapping:
        if self._grid_mapping is None:
            with self._lock:
                self._grid_mapping = self._get_grid_mapping_lazily()
        return self._grid_mapping

    @property
    def tile_grid(self) -> TileGrid:
        if self._tile_grid is None:
            with self._lock:
                self._tile_grid = self._get_tile_grid_lazily()
        return self._tile_grid

    @property
    def lock(self) -> threading.RLock:
        """
        Get the reentrant lock used by this object to synchronize
        lazy instantiation of properties.
        """
        return self._lock

    def get_dataset(self, index: int) -> xr.Dataset:
        """
        Get or compute the dataset for the level at given *index*.

        :param index: the level index
        :return: the dataset for the level at *index*.
        """
        if index not in self._level_datasets:
            with self._lock:
                # noinspection PyTypeChecker
                level_dataset = self._get_dataset_lazily(index,
                                                         self._parameters)
                self.set_dataset(index, level_dataset)
        # noinspection PyTypeChecker
        return self._level_datasets[index]

    def set_dataset(self, index: int, level_dataset: xr.Dataset):
        """
        Set the dataset for the level at given *index*.

        Callers need to ensure that the given *level_dataset*
        has the correct spatial dimension sizes for the
        given level at *index*.

        :param index: the level index
        :param level_dataset: the dataset for the level at *index*.
        """
        with self._lock:
            self._level_datasets[index] = level_dataset

    @abstractmethod
    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        """
        Retrieve, i.e. read or compute, the dataset for the level at given *index*.

        :param index: the level index
        :param parameters: *parameters* keyword argument that was passed to constructor.
        :return: the dataset for the level at *index*.
        """

    def _get_grid_mapping_lazily(self) -> GridMapping:
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return GridMapping.from_dataset(self.get_dataset(0))

    def _get_tile_grid_lazily(self) -> TileGrid:
        """
        Retrieve, i.e. read or compute, the tile grid used by
        the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return self.grid_mapping.tile_grid

    def close(self):
        with self._lock:
            for dataset in self._level_datasets.values():
                if dataset is not None:
                    dataset.close()


class CombinedMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset that is a combination of other multi-level datasets.

    :param ml_datasets: The multi-level datasets to be combined. At least two must be provided.
    :param ds_id: Optional dataset identifier.
    :param combiner_function: A function used to combine the datasets. It receives a list of
        datasets (``xarray.Dataset`` instances) and *combiner_params* as keyword arguments.
        Defaults to function ``xarray.merge()`` with default parameters.
    :param combiner_params: Parameters to the *combiner_function* passed as keyword arguments.
    """

    def __init__(self,
                 ml_datasets: Sequence[MultiLevelDataset],
                 tile_grid: TileGrid = None,
                 ds_id: str = None,
                 combiner_function: Callable = None,
                 combiner_params: Dict[str, Any] = None):
        super().__init__(tile_grid=tile_grid, ds_id=ds_id, parameters=combiner_params)
        if not ml_datasets or len(ml_datasets) < 2:
            raise ValueError('ml_datasets must have at least two elements')
        self._ml_datasets = ml_datasets
        self._combiner_function = combiner_function or xr.merge

    def _get_dataset_lazily(self, index: int, combiner_params: Dict[str, Any]) -> xr.Dataset:
        datasets = [ml_dataset.get_dataset(index) for ml_dataset in self._ml_datasets]
        return self._combiner_function(datasets, **combiner_params)

    def close(self):
        for ml_dataset in self._ml_datasets:
            ml_dataset.close()


class MappedMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(self,
                 ml_dataset: MultiLevelDataset,
                 mapper_function: Callable[[xr.Dataset, Dict[str, Any]], xr.Dataset],
                 tile_grid: TileGrid = None,
                 ds_id: str = None,
                 mapper_params: Dict[str, Any] = None):
        super().__init__(tile_grid=tile_grid, ds_id=ds_id, parameters=mapper_params)
        self._ml_dataset = ml_dataset
        self._mapper_function = mapper_function

    def _get_dataset_lazily(self, index: int, mapper_params: Dict[str, Any]) -> xr.Dataset:
        return self._mapper_function(self._ml_dataset.get_dataset(index), **mapper_params)

    def close(self):
        self._ml_dataset.close()


class IdentityMultiLevelDataset(MappedMultiLevelDataset):
    def __init__(self, ml_dataset: MultiLevelDataset, ds_id: str = None):
        super().__init__(ml_dataset, lambda ds: ds, ds_id=ds_id)


class FileStorageMultiLevelDataset(LazyMultiLevelDataset):
    """
    A stored multi-level dataset whose level datasets are lazily read from storage location.

    :param dir_path: The directory containing the level datasets.
    :param zarr_kwargs: Keyword arguments accepted by the ``xarray.open_zarr()`` function.
    :param ds_id: Optional dataset identifier.
    """

    def __init__(self,
                 dir_path: str,
                 ds_id: str = None,
                 zarr_kwargs: Dict[str, Any] = None,
                 exception_type: type = ValueError):
        file_paths = os.listdir(dir_path)
        level_paths = {}
        num_levels = -1
        for filename in file_paths:
            file_path = os.path.join(dir_path, filename)
            basename, ext = os.path.splitext(filename)
            if basename.isdigit():
                index = int(basename)
                num_levels = max(num_levels, index + 1)
                if os.path.isfile(file_path) and ext == ".link":
                    level_paths[index] = (ext, file_path)
                elif os.path.isdir(file_path) and ext == ".zarr":
                    level_paths[index] = (ext, file_path)

        if num_levels != len(level_paths):
            raise exception_type(f"Inconsistent levels directory:"
                                 f" expected {num_levels} but found {len(level_paths)} entries:"
                                 f" {dir_path}")

        super().__init__(ds_id=ds_id, parameters=zarr_kwargs)
        self._dir_path = dir_path
        self._level_paths = level_paths
        self._num_levels = num_levels

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        """
        Read the dataset for the level at given *index*.

        :param index: the level index
        :param parameters: keyword arguments passed to xr.open_zarr()
        :return: the dataset for the level at *index*.
        """
        ext, level_path = self._level_paths[index]
        if ext == ".link":
            with open(level_path, "r") as fp:
                level_path = fp.read()
                # if file_path is a relative path, resolve it against the levels directory
                if not os.path.isabs(level_path):
                    base_dir = os.path.dirname(self._dir_path)
                    level_path = os.path.join(base_dir, level_path)
        with measure_time(tag=f"opened local dataset {level_path} for level {index}"):
            return assert_cube(xr.open_zarr(level_path, **parameters), name=level_path)

    def _get_tile_grid_lazily(self) -> TileGrid:
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        tile_grid = self.grid_mapping.tile_grid
        if tile_grid.num_levels != self._num_levels:
            raise ValueError(f'Detected inconsistent'
                             f' number of detail levels,'
                             f' expected {tile_grid.num_levels},'
                             f' found {self._num_levels}.')
        return tile_grid


class ObjectStorageMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are lazily read from object storage locations.

    :param dir_path: The directory containing the level datasets.
    :param zarr_kwargs: Keyword arguments accepted by the ``xarray.open_zarr()`` function.
    :param ds_id: Optional dataset identifier.
    """

    def __init__(self,
                 s3_file_system: s3fs.S3FileSystem,
                 dir_path: str,
                 zarr_kwargs: Dict[str, Any] = None,
                 ds_id: str = None,
                 chunk_cache_capacity: int = None,
                 exception_type: type = ValueError):

        level_paths = {}
        entries = s3_file_system.ls(dir_path, detail=False)
        for entry in entries:
            level_dir = entry.split("/")[-1]
            basename, ext = os.path.splitext(level_dir)
            if basename.isdigit():
                level = int(basename)
                if entry.endswith(".zarr") and s3_file_system.isdir(entry):
                    level_paths[level] = (ext, dir_path + "/" + level_dir)
                elif entry.endswith(".link") and s3_file_system.isfile(entry):
                    level_paths[level] = (ext, dir_path + "/" + level_dir)

        num_levels = len(level_paths)
        # Consistency check
        for level in range(num_levels):
            if level not in level_paths:
                raise exception_type(f"Invalid multi-level dataset {ds_id!r}: missing level {level} in {dir_path}")

        super().__init__(ds_id=ds_id, parameters=zarr_kwargs)
        self._s3_file_system = s3_file_system
        self._dir_path = dir_path
        self._level_paths = level_paths
        self._num_levels = num_levels

        self._chunk_cache_capacities = None
        if chunk_cache_capacity:
            weights = []
            weigth_sum = 0
            for level in range(num_levels):
                weight = 2 ** (num_levels - 1 - level)
                weight *= weight
                weigth_sum += weight
                weights.append(weight)
            self._chunk_cache_capacities = [round(chunk_cache_capacity * weight / weigth_sum)
                                            for weight in weights]

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def get_chunk_cache_capacity(self, index: int) -> Optional[int]:
        """
        Get the chunk cache capacity for given level.

        :param index: The level index.
        :return: The chunk cache capacity for given level or None.
        """
        return self._chunk_cache_capacities[index] if self._chunk_cache_capacities else None

    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        """
        Read the dataset for the level at given *index*.

        :param index: the level index
        :param parameters: keyword arguments passed to xr.open_zarr()
        :return: the dataset for the level at *index*.
        """
        ext, level_path = self._level_paths[index]
        if ext == ".link":
            with self._s3_file_system.open(level_path, "w") as fp:
                level_path = fp.read()
                # if file_path is a relative path, resolve it against the levels directory
                if not os.path.isabs(level_path):
                    base_dir = os.path.dirname(self._dir_path)
                    level_path = os.path.join(base_dir, level_path)
        store = s3fs.S3Map(root=level_path, s3=self._s3_file_system, check=False)
        max_size = self.get_chunk_cache_capacity(index)
        if max_size:
            store = zarr.LRUStoreCache(store, max_size=max_size)
        with measure_time(tag=f"opened remote dataset {level_path} for level {index}"):
            consolidated = self._s3_file_system.exists(f'{level_path}/.zmetadata')
            return assert_cube(xr.open_zarr(store, consolidated=consolidated, **parameters), name=level_path)

    def _get_tile_grid_lazily(self) -> TileGrid:
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        tile_grid = self.grid_mapping.tile_grid
        if tile_grid.num_levels != self._num_levels:
            raise ValueError(f'Detected inconsistent'
                             f' number of detail levels,'
                             f' expected {tile_grid.num_levels},'
                             f' found {self._num_levels}.')
        return tile_grid


class BaseMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are a created by down-sampling a base dataset.

    :param base_dataset: The base dataset for the level at index zero.
    :param ds_id: Optional dataset identifier.
    """

    def __init__(self,
                 base_dataset: xr.Dataset,
                 tile_grid: TileGrid = None,
                 ds_id: str = None):
        assert_instance(base_dataset, xr.Dataset, name='base_dataset')
        self._base_cube, grid_mapping, _ = decode_cube(
            base_dataset,
            force_non_empty=True
        )
        super().__init__(grid_mapping=grid_mapping,
                         tile_grid=tile_grid,
                         ds_id=ds_id)

    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        """
        Compute the dataset at level *index*: If *index* is zero, return the base image passed to constructor,
        otherwise down-sample the dataset for the level at given *index*.

        :param index: the level index
        :param parameters: currently unused
        :return: the dataset for the level at *index*.
        """
        if index == 0:
            level_dataset = self._base_cube
        else:
            base_dataset = self._base_cube
            step = 2 ** index
            data_vars = {}
            for var_name in base_dataset.data_vars:
                var = base_dataset[var_name]
                var = var[..., ::step, ::step]
                data_vars[var_name] = var
            level_dataset = xr.Dataset(data_vars,
                                       attrs=self._base_cube.attrs)

        # Tile each level according to grid mapping
        tile_size = self.grid_mapping.tile_size
        if tile_size is not None:
            level_dataset, _ = rechunk_cube(level_dataset,
                                            self.grid_mapping,
                                            tile_size=tile_size)
        return level_dataset


# TODO (forman): rename to ScriptedMultiLevelDataset
# TODO (forman): use new xcube.core.byoa package here

class ComputedMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are a computed from a Python script.
    """

    def __init__(self,
                 script_path: str,
                 callable_name: str,
                 input_ml_dataset_ids: Sequence[str],
                 input_ml_dataset_getter: Callable[[str], MultiLevelDataset],
                 input_parameters: Mapping[str, Any],
                 ds_id: str = None,
                 exception_type: type = ValueError):

        input_parameters = input_parameters or {}
        super().__init__(ds_id=ds_id, parameters=input_parameters)

        try:
            with open(script_path) as fp:
                python_code = fp.read()
        except OSError as e:
            raise exception_type(
                f"Failed to read Python code for in-memory dataset {ds_id!r} from {script_path!r}: {e}") from e

        global_env = dict()
        try:
            exec(python_code, global_env, None)
        except Exception as e:
            raise exception_type(f"Failed to compute in-memory dataset {ds_id!r} from {script_path!r}: {e}") from e

        if not callable_name or not callable_name.isidentifier():
            raise exception_type(f"Invalid dataset descriptor {ds_id!r}: "
                                 f"{callable_name!r} is not a valid Python identifier")
        callable_obj = global_env.get(callable_name)
        if callable_obj is None:
            raise exception_type(f"Invalid in-memory dataset descriptor {ds_id!r}: "
                                 f"no callable named {callable_name!r} found in {script_path!r}")
        if not callable(callable_obj):
            raise exception_type(f"Invalid in-memory dataset descriptor {ds_id!r}: "
                                 f"object {callable_name!r} in {script_path!r} is not callable")

        if not callable_name or not callable_name.isidentifier():
            raise exception_type(f"Invalid in-memory dataset descriptor {ds_id!r}: "
                                 f"{callable_name!r} is not a valid Python identifier")
        if not input_ml_dataset_ids:
            raise exception_type(f"Invalid in-memory dataset descriptor {ds_id!r}: "
                                 f"Input dataset(s) missing for callable {callable_name!r}")
        for input_param_name in input_parameters.keys():
            if not input_param_name or not input_param_name.isidentifier():
                raise exception_type(f"Invalid in-memory dataset descriptor {ds_id!r}: "
                                     f"Input parameter {input_param_name!r} for callable {callable_name!r} "
                                     f"is not a valid Python identifier")
        self._callable_name = callable_name
        self._callable_obj = callable_obj
        self._input_ml_dataset_ids = input_ml_dataset_ids
        self._input_ml_dataset_getter = input_ml_dataset_getter
        self._exception_type = exception_type

    def _get_tile_grid_lazily(self) -> TileGrid:
        return self._input_ml_dataset_getter(self._input_ml_dataset_ids[0]).tile_grid

    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        input_datasets = [self._input_ml_dataset_getter(ds_id).get_dataset(index)
                          for ds_id in self._input_ml_dataset_ids]
        try:
            with measure_time(tag=f"computed in-memory dataset {self.ds_id!r} at level {index}"):
                computed_value = self._callable_obj(*input_datasets, **parameters)
        except Exception as e:
            raise self._exception_type(f"Failed to compute in-memory dataset {self.ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: {e}") from e
        if not isinstance(computed_value, xr.Dataset):
            raise self._exception_type(f"Failed to compute in-memory dataset {self.ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: "
                                       f"expected an xarray.Dataset but got {type(computed_value)}")
        return assert_cube(computed_value, name=self.ds_id)


def guess_ml_dataset_format(path: str) -> str:
    """
    Guess a multilevel-dataset format for a file system path or URL given by *path*.

    :param path: A file system path or URL.
    :return: The name of a dataset format guessed from *path*.
    """
    if path.endswith('.levels'):
        return FORMAT_NAME_LEVELS
    if path.endswith('.py'):
        return FORMAT_NAME_SCRIPT
    return guess_dataset_format(path)


def open_ml_dataset(path: str,
                    ds_id: str = None,
                    exception_type: type = ValueError,
                    **kwargs) -> MultiLevelDataset:
    """
    Open a multi-level dataset.

    :param path: dataset path
    :param ds_id: Optional dataset ID, if not given, a new UUID will be generated.
    :param exception_type: The type of exception to be thrown, defaults to ValueError
    :param kwargs: format specific parameters, e.g, "endpoint_url", "region_name"
    :return: a multi-level dataset
    """
    if not path:
        raise ValueError('path must be given')
    if is_s3_url(path):
        return open_ml_dataset_from_object_storage(path, ds_id=ds_id, exception_type=exception_type, **kwargs)
    elif path.endswith('.py'):
        return open_ml_dataset_from_python_code(path, ds_id=ds_id, exception_type=exception_type, **kwargs)
    else:
        return open_ml_dataset_from_local_fs(path, ds_id=ds_id, exception_type=exception_type, **kwargs)


# noinspection PyUnusedLocal
def open_ml_dataset_from_object_storage(path: str,
                                        data_format: str = None,
                                        ds_id: str = None,
                                        exception_type: type = ValueError,
                                        s3_kwargs: Mapping[str, Any] = None,
                                        s3_client_kwargs: Mapping[str, Any] = None,
                                        chunk_cache_capacity: int = None,
                                        **kwargs) -> MultiLevelDataset:
    data_format = data_format or guess_ml_dataset_format(path)

    s3, root = parse_s3_fs_and_root(path,
                                    s3_kwargs=s3_kwargs,
                                    s3_client_kwargs=s3_client_kwargs,
                                    mode='r')

    if data_format == FORMAT_NAME_ZARR:
        store = s3fs.S3Map(root=root, s3=s3, check=False)
        if chunk_cache_capacity:
            store = zarr.LRUStoreCache(store, max_size=chunk_cache_capacity)
        with measure_time(tag=f"opened remote zarr dataset {path}"):
            consolidated = s3.exists(f'{root}/.zmetadata')
            ds = assert_cube(xr.open_zarr(store, consolidated=consolidated, **kwargs))
        return BaseMultiLevelDataset(ds, ds_id=ds_id)
    elif data_format == FORMAT_NAME_LEVELS:
        with measure_time(tag=f"opened remote levels dataset {path}"):
            return ObjectStorageMultiLevelDataset(s3,
                                                  root,
                                                  zarr_kwargs=kwargs,
                                                  ds_id=ds_id,
                                                  chunk_cache_capacity=chunk_cache_capacity,
                                                  exception_type=exception_type)

    raise exception_type(f'Unrecognized multi-level dataset format {data_format!r} for path {path!r}')


def open_ml_dataset_from_local_fs(path: str,
                                  data_format: str = None,
                                  ds_id: str = None,
                                  exception_type: type = ValueError,
                                  **kwargs) -> MultiLevelDataset:
    data_format = data_format or guess_ml_dataset_format(path)

    if data_format == FORMAT_NAME_NETCDF4:
        with measure_time(tag=f"opened local NetCDF dataset {path}"):
            ds = assert_cube(xr.open_dataset(path, **kwargs))
            return BaseMultiLevelDataset(ds, ds_id=ds_id)
    elif data_format == FORMAT_NAME_ZARR:
        with measure_time(tag=f"opened local zarr dataset {path}"):
            ds = assert_cube(xr.open_zarr(path, **kwargs))
            return BaseMultiLevelDataset(ds, ds_id=ds_id)
    elif data_format == FORMAT_NAME_LEVELS:
        with measure_time(tag=f"opened local levels dataset {path}"):
            return FileStorageMultiLevelDataset(path, ds_id=ds_id, zarr_kwargs=kwargs)

    raise exception_type(f'Unrecognized multi-level dataset format {data_format!r} for path {path!r}')


def open_ml_dataset_from_python_code(script_path: str,
                                     callable_name: str,
                                     input_ml_dataset_ids: Sequence[str] = None,
                                     input_ml_dataset_getter: Callable[[str], MultiLevelDataset] = None,
                                     input_parameters: Mapping[str, Any] = None,
                                     ds_id: str = None,
                                     exception_type: type = ValueError) -> MultiLevelDataset:
    with measure_time(tag=f"opened memory dataset {script_path}"):
        return ComputedMultiLevelDataset(script_path,
                                         callable_name,
                                         input_ml_dataset_ids,
                                         input_ml_dataset_getter,
                                         input_parameters,
                                         ds_id=ds_id,
                                         exception_type=exception_type)


def augment_ml_dataset(ml_dataset: MultiLevelDataset,
                       script_path: str,
                       callable_name: str,
                       input_ml_dataset_getter: Callable[[str], MultiLevelDataset],
                       input_ml_dataset_setter: Callable[[MultiLevelDataset], None],
                       input_parameters: Mapping[str, Any] = None,
                       exception_type: type = ValueError):
    with measure_time(tag=f"added augmentation from {script_path}"):
        orig_id = ml_dataset.ds_id
        aug_id = uuid.uuid4()
        aug_inp_id = f'aug-input-{aug_id}'
        aug_inp_ds = IdentityMultiLevelDataset(ml_dataset, ds_id=aug_inp_id)
        input_ml_dataset_setter(aug_inp_ds)
        aug_ds = ComputedMultiLevelDataset(script_path,
                                           callable_name,
                                           [aug_inp_id],
                                           input_ml_dataset_getter,
                                           input_parameters,
                                           ds_id=f'aug-{aug_id}',
                                           exception_type=exception_type)
        return CombinedMultiLevelDataset([ml_dataset, aug_ds], ds_id=orig_id)


def write_levels(ml_dataset: MultiLevelDataset,
                 levels_path: str,
                 s3_kwargs: Dict[str, Any] = None,
                 s3_client_kwargs: Dict[str, Any] = None):
    tile_w, tile_h = ml_dataset.tile_grid.tile_size
    chunks = dict(time=1, lat=tile_h, lon=tile_w)
    for level in range(ml_dataset.num_levels):
        level_dataset = ml_dataset.get_dataset(level)
        level_dataset = level_dataset.chunk(chunks)
        print(f'writing level {level + 1}...')
        write_cube(level_dataset,
                   f'{levels_path}/{level}.zarr',
                   'zarr',
                   s3_kwargs=s3_kwargs,
                   s3_client_kwargs=s3_client_kwargs)
        print(f'written level {level + 1}')
