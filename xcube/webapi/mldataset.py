import os
import threading
from abc import abstractmethod, ABCMeta
from typing import Sequence, Any, Dict, Callable

import s3fs
import xarray as xr
import zarr

from .im import TileGrid
from .perf import measure_time
from .utils import get_dataset_bounds


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
    def tile_grid(self) -> TileGrid:
        """
        :return: the tile grid.
        """

    @property
    def num_levels(self) -> int:
        """
        :return: the number of pyramid levels.
        """
        return self.tile_grid.num_levels

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


class LazyMultiLevelDataset(MultiLevelDataset, metaclass=ABCMeta):
    """
    A multi-level dataset where each level dataset is lazily retrieved, i.e. read or computed by the abstract method
    ``get_dataset_lazily(index, **kwargs)``.

    If no *tile_grid* is passed it will be retrieved lazily by using the ``get_tile_grid_lazily()`` method,
    which may be overridden.The default implementation computes a new tile grid based on the dataset at level zero.

    :param tile_grid: The tile grid. If None, a new tile grid will be computed based on the dataset at level zero.
    :param kwargs: Extra keyword arguments that will be passed to the ``get_dataset_lazily`` method.
    """

    def __init__(self, tile_grid: TileGrid = None, kwargs: Dict[str, Any] = None):
        self._tile_grid = tile_grid
        self._level_datasets = {}
        self._kwargs = kwargs
        self._lock = threading.RLock()

    @property
    def tile_grid(self) -> TileGrid:
        if self._tile_grid is None:
            with self._lock:
                self._tile_grid = self._get_tile_grid_lazily()
        return self._tile_grid

    def get_dataset(self, index: int) -> xr.Dataset:
        """
        Get or compute the dataset for the level at given *index*.

        :param index: the level index
        :return: the dataset for the level at *index*.
        """
        if index not in self._level_datasets:
            kwargs = self._kwargs if self._kwargs is not None else {}
            with self._lock:
                # noinspection PyTypeChecker
                self._level_datasets[index] = self._get_dataset_lazily(index, **kwargs)
        # noinspection PyTypeChecker
        return self._level_datasets[index]

    @abstractmethod
    def _get_dataset_lazily(self, index: int, **kwargs) -> xr.Dataset:
        """
        Retrieve, i.e. read or compute, the dataset for the level at given *index*.

        :param index: the level index
        :param kwargs: Extra keyword arguments passed to constructor.
        :return: the dataset for the level at *index*.
        """

    def _get_tile_grid_lazily(self):
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return _get_dataset_tile_grid(self.get_dataset(0))

    def close(self):
        with self._lock:
            for dataset in self._level_datasets.values():
                if dataset is not None:
                    dataset.close()


class FileStorageMultiLevelDataset(LazyMultiLevelDataset):
    """
    A stored multi-level dataset whose level datasets are lazily read from storage location.

    :param dir_path: The directory containing the level datasets.
    :param zarr_kwargs: Keyword arguments accepted by the ``xarray.open_zarr()`` function.
    """

    def __init__(self, dir_path: str, zarr_kwargs: Dict[str, Any] = None):
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
            raise ValueError(f"Inconsistent levels directory:"
                             f" expected {num_levels} but found {len(level_paths)} entries:"
                             f" {dir_path}")

        super().__init__(kwargs=zarr_kwargs)
        self._dir_path = dir_path
        self._level_paths = level_paths
        self._num_levels = num_levels

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def _get_dataset_lazily(self, index: int, **zarr_kwargs) -> xr.Dataset:
        """
        Read the dataset for the level at given *index*.

        :param index: the level index
        :param zarr_kwargs: kwargs passed to xr.open_zarr()
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
            return xr.open_zarr(level_path, **zarr_kwargs)

    def _get_tile_grid_lazily(self):
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return _get_dataset_tile_grid(self.get_dataset(0), num_levels=self._num_levels)


class ObjectStorageMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are lazily read from object storage locations.

    :param dir_path: The directory containing the level datasets.
    :param zarr_kwargs: Keyword arguments accepted by the ``xarray.open_zarr()`` function.
    """

    def __init__(self, ds_id: str,
                 obs_file_system: s3fs.S3FileSystem,
                 dir_path: str,
                 zarr_kwargs: Dict[str, Any] = None, exception_type=ValueError):

        level_paths = {}
        for entry in obs_file_system.walk(dir_path, directories=True):
            basename = None
            if entry.endswith(".zarr") and obs_file_system.isdir(entry):
                basename, _ = os.path.splitext(entry)
            elif entry.endswith(".link") and obs_file_system.isfile(entry):
                basename, _ = os.path.splitext(entry)
            if basename is not None and basename.isdigit():
                level = int(basename)
                level_paths[level] = dir_path + "/" + entry

        num_levels = len(level_paths)
        # Consistency check
        for level in range(num_levels):
            if level not in level_paths:
                raise exception_type(f"Invalid dataset descriptor {ds_id!r}: missing level {level} in {dir_path}")

        super().__init__(kwargs=zarr_kwargs)
        self._obs_file_system = obs_file_system
        self._dir_path = dir_path
        self._level_paths = level_paths
        self._num_levels = num_levels

    @property
    def num_levels(self) -> int:
        return self._num_levels

    def _get_dataset_lazily(self, index: int, **zarr_kwargs) -> xr.Dataset:
        """
        Read the dataset for the level at given *index*.

        :param index: the level index
        :param zarr_kwargs: kwargs passed to xr.open_zarr()
        :return: the dataset for the level at *index*.
        """
        ext, level_path = self._level_paths[index]
        if ext == ".link":
            with self._obs_file_system.open(level_path, "w") as fp:
                level_path = fp.read()
                # if file_path is a relative path, resolve it against the levels directory
                if not os.path.isabs(level_path):
                    base_dir = os.path.dirname(self._dir_path)
                    level_path = os.path.join(base_dir, level_path)

        store = s3fs.S3Map(root=level_path, s3=self._obs_file_system, check=False)
        cached_store = zarr.LRUStoreCache(store, max_size=2 ** 28)
        with measure_time(tag=f"opened remote dataset {level_path} for level {index}"):
            return xr.open_zarr(cached_store, **zarr_kwargs)

    def _get_tile_grid_lazily(self):
        """
        Retrieve, i.e. read or compute, the tile grid used by the multi-level dataset.

        :return: the dataset for the level at *index*.
        """
        return _get_dataset_tile_grid(self.get_dataset(0), num_levels=self._num_levels)


class BaseMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are a created by down-sampling a base dataset.

    :param base_dataset: The base dataset for the level at index zero.
    """

    def __init__(self, base_dataset: xr.Dataset, tile_grid: TileGrid = None):
        super().__init__(tile_grid=tile_grid)
        if base_dataset is None:
            raise ValueError("base_dataset must be given")
        self._base_dataset = base_dataset

    def _get_dataset_lazily(self, index: int, **kwargs) -> xr.Dataset:
        """
        Compute the dataset at level *index*: If *index* is zero, return the base image passed to constructor,
        otherwise down-sample the dataset for the level at given *index*.

        :param index: the level index
        :param kwargs: currently unused
        :return: the dataset for the level at *index*.
        """
        if index == 0:
            level_dataset = self._base_dataset
        else:
            base_dataset = self._base_dataset
            step = 2 ** index
            data_vars = {}
            for var_name in base_dataset.data_vars:
                var = base_dataset[var_name]
                var = var[..., ::step, ::step]
                data_vars[var_name] = var
            level_dataset = xr.Dataset(data_vars, attrs=base_dataset.attrs)
        return level_dataset


class ComputedMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are a computed from the levels of other multi-level datasets.
    """

    def __init__(self,
                 ds_id: str,
                 script_path: str,
                 callable_name: str,
                 input_ml_dataset_ids: Sequence[str],
                 input_ml_dataset_getter: Callable[[str], MultiLevelDataset],
                 input_parameters: Dict[str, Any],
                 exception_type=ValueError):
        super().__init__(kwargs=input_parameters)

        try:
            with open(script_path) as fp:
                python_code = fp.read()
        except OSError as e:
            raise exception_type(
                f"Failed to read Python code for in-memory dataset {ds_id!r} from {script_path!r}: {e}") from e

        local_env = dict()
        global_env = None
        try:
            exec(python_code, global_env, local_env)
        except Exception as e:
            raise exception_type(f"Failed to compute in-memory dataset {ds_id!r} from {script_path!r}: {e}") from e

        if not callable_name or not callable_name.isidentifier():
            raise exception_type(f"Invalid dataset descriptor {ds_id!r}: "
                                 f"{callable_name!r} is not a valid Python identifier")
        callable_obj = local_env.get(callable_name)
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
        self._ds_id = ds_id
        self._callable_name = callable_name
        self._callable_obj = callable_obj
        self._input_ml_dataset_ids = input_ml_dataset_ids
        self._input_ml_dataset_getter = input_ml_dataset_getter
        self._exception_type = exception_type

    def _get_tile_grid_lazily(self) -> TileGrid:
        return self._input_ml_dataset_getter(self._input_ml_dataset_ids[0]).tile_grid

    def _get_dataset_lazily(self, index: int, **kwargs) -> xr.Dataset:
        input_datasets = [self._input_ml_dataset_getter(ds_id).get_dataset(index)
                          for ds_id in self._input_ml_dataset_ids]
        try:
            with measure_time(tag=f"computed in-memory dataset {self._ds_id!r} at level {index}"):
                computed_value = self._callable_obj(*input_datasets, **kwargs)
        except Exception as e:
            raise self._exception_type(f"Failed to compute in-memory dataset {self._ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: {e}") from e
        if not isinstance(computed_value, xr.Dataset):
            raise self._exception_type(f"Failed to compute in-memory dataset {self._ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: "
                                       f"expected an xarray.Dataset but got {type(computed_value)}")
        return computed_value


def _get_dataset_tile_grid(dataset: xr.Dataset, num_levels: int = None):
    geo_extent = get_dataset_bounds(dataset)
    inv_y = float(dataset.lat[0]) < float(dataset.lat[-1])
    width, height, tile_width, tile_height = _get_cube_spatial_sizes(dataset)
    if num_levels is not None and tile_width is not None and tile_height is not None:
        width_0 = width
        height_0 = height
        for i in range(num_levels):
            width_0 = (width_0 + 1) // 2
            height_0 = (height_0 + 1) // 2
        num_level_zero_tiles_x = (width_0 + tile_width - 1) // tile_width
        num_level_zero_tiles_y = (height_0 + tile_height - 1) // tile_height
        tile_grid = TileGrid(num_levels,
                             num_level_zero_tiles_x, num_level_zero_tiles_y,
                             tile_width, tile_height,
                             geo_extent, inv_y)
    else:
        try:
            tile_grid = TileGrid.create(width, height,
                                        tile_width, tile_height,
                                        geo_extent, inv_y)
        except ValueError:
            num_levels = 1
            num_level_zero_tiles_x = 1
            num_level_zero_tiles_y = 1
            tile_grid = TileGrid(num_levels,
                                 num_level_zero_tiles_x, num_level_zero_tiles_y,
                                 width, height, geo_extent, inv_y)

    return tile_grid


def _get_cube_spatial_sizes(dataset: xr.Dataset):
    first_var_name = None
    spatial_shape = None
    spatial_chunks = None
    for var_name in dataset.data_vars:
        var = dataset[var_name]

        if var.ndim < 2 or var.dims[-2:] != ("lat", "lon"):
            continue

        if first_var_name is None:
            first_var_name = var_name

        if spatial_shape is None:
            spatial_shape = var.shape[-2:]
        elif spatial_shape != var.shape[-2:]:
            raise ValueError(f"variables in dataset have different spatial shapes:"
                             f" variable {first_var_name!r} has {spatial_shape}"
                             f" while {var_name!r} has {var.shape}")

        if var.chunks is not None:
            if spatial_chunks is None:
                spatial_chunks = var.chunks[-2:]
            elif spatial_chunks != var.chunks[-2:]:
                raise ValueError(f"variables in dataset have different spatial chunks:"
                                 f" variable {first_var_name!r} has {spatial_chunks}"
                                 f" while {var_name!r} has {var.chunks}")

    if spatial_shape is None:
        raise ValueError("no variables with spatial dimensions found")

    width, height = spatial_shape[-1], spatial_shape[-2]
    tile_width, tile_height = None, None

    if spatial_chunks is not None:
        def to_int(v):
            return v if isinstance(v, int) else v[0]

        spatial_chunks = tuple(map(to_int, spatial_chunks))
        tile_width, tile_height = spatial_chunks[-1], spatial_chunks[-2]

    return width, height, tile_width, tile_height
