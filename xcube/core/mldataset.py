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

import math
import os.path
import threading
import sys
import uuid
from abc import abstractmethod, ABCMeta
from functools import cached_property
from typing import Sequence, Any, Dict, Callable, Mapping, Optional, Tuple

import xarray as xr

from xcube.constants import FORMAT_NAME_LEVELS
from xcube.constants import FORMAT_NAME_SCRIPT
from xcube.constants import LOG
from xcube.core.dsio import guess_dataset_format
from xcube.core.gridmapping import GridMapping
from xcube.core.schema import rechunk_cube
from xcube.core.subsampling import AggMethods
from xcube.core.subsampling import assert_valid_agg_methods
from xcube.core.subsampling import subsample_dataset
from xcube.core.tilingscheme import TilingScheme
from xcube.core.tilingscheme import get_num_levels
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.perf import measure_time
from xcube.util.types import ScalarOrPair
from xcube.util.types import normalize_scalar_or_pair

_DEPRECATED_OPEN_ML_DATASET = ('Use xcube data store framework'
                               ' to open multi-level datasets.')
_DEPRECATED_WRITE_ML_DATASET = ('Use xcube data store framework'
                                ' to write multi-level datasets.')

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

    @ds_id.setter
    @abstractmethod
    def ds_id(self, ds_id: str):
        """
        Set the dataset identifier.
        """

    @property
    @abstractmethod
    def grid_mapping(self) -> GridMapping:
        """
        :return: the CF-conformal grid mapping
        """

    @property
    @abstractmethod
    def num_levels(self) -> int:
        """
        :return: the number of pyramid levels.
        """

    @cached_property
    def resolutions(self) -> Sequence[Tuple[float, float]]:
        """
        :return: the x,y resolutions for each level given in the
            spatial units of the dataset's CRS
            (i.e. ``self.grid_mapping.crs``).
        """
        x_res_0, y_res_0 = self.grid_mapping.xy_res
        return [(x_res_0 * (1 << level), y_res_0 * (1 << level))
                for level in range(self.num_levels)]

    @cached_property
    def avg_resolutions(self) -> Sequence[float]:
        """
        :return: the average x,y resolutions for each level given in the
            spatial units of the dataset's CRS
            (i.e. ``self.grid_mapping.crs``).
        """
        x_res_0, y_res_0 = self.grid_mapping.xy_res
        xy_res_0 = math.sqrt(x_res_0 * y_res_0)
        return [xy_res_0 * (1 << level)
                for level in range(self.num_levels)]

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
              ds_id: str = None) -> 'MultiLevelDataset':
        """ Apply function to all level datasets and return a new multi-level dataset."""
        return MappedMultiLevelDataset(self, function,
                                       ds_id=ds_id,
                                       mapper_params=kwargs)

    def derive_tiling_scheme(self, tiling_scheme: TilingScheme):
        """
        Derive a new tiling scheme for the given one with defined
        minimum and maximum level indices.
        """
        min_level, max_level = tiling_scheme.get_levels_for_resolutions(
            self.avg_resolutions,
            self.grid_mapping.spatial_unit_name
        )
        return tiling_scheme.derive(min_level=min_level,
                                    max_level=max_level)

    def get_level_for_resolution(self, xy_res: ScalarOrPair[float]) -> int:
        """
        Get the index of the level that best represents the given resolution.

        :param xy_res: the resolution in x- and y-direction
        :return: a level ranging from 0 to self.num_levels - 1
        """
        given_x_res, given_y_res = normalize_scalar_or_pair(xy_res,
                                                            item_type=float)
        for level, (x_res, y_res) in enumerate(self.resolutions):
            if x_res > given_x_res and y_res > given_y_res:
                return max(0, level - 1)
        return self.num_levels - 1


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
                 grid_mapping: Optional[GridMapping] = None,
                 num_levels: Optional[int] = None,
                 ds_id: Optional[str] = None,
                 parameters: Optional[Mapping[str, Any]] = None):
        if grid_mapping is not None:
            assert_instance(grid_mapping, GridMapping, name='grid_mapping')
        if ds_id is not None:
            assert_instance(ds_id, str, name='ds_id')
        self._grid_mapping = grid_mapping
        self._num_levels = num_levels
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

    @ds_id.setter
    def ds_id(self, ds_id: str):
        assert_instance(ds_id, str, name='ds_id')
        self._ds_id = ds_id

    @property
    def grid_mapping(self) -> GridMapping:
        if self._grid_mapping is None:
            with self._lock:
                self._grid_mapping = self._get_grid_mapping_lazily()
        return self._grid_mapping

    @property
    def num_levels(self) -> int:
        if self._num_levels is None:
            with self._lock:
                self._num_levels = self._get_num_levels_lazily()
        return self._num_levels

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
    def _get_num_levels_lazily(self) -> int:
        """
        Retrieve, i.e. read or compute, the number of levels.

        :return: the number of dataset levels.
        """

    @abstractmethod
    def _get_dataset_lazily(self, index: int,
                            parameters: Dict[str, Any]) -> xr.Dataset:
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
                 ds_id: str = None,
                 combiner_function: Callable = None,
                 combiner_params: Dict[str, Any] = None):
        if not ml_datasets or len(ml_datasets) < 2:
            raise ValueError('ml_datasets must have at least two elements')
        super().__init__(ds_id=ds_id,
                         parameters=combiner_params)
        self._ml_datasets = ml_datasets
        self._combiner_function = combiner_function or xr.merge

    def _get_num_levels_lazily(self) -> int:
        return self._ml_datasets[0].num_levels

    def _get_dataset_lazily(self, index: int,
                            combiner_params: Dict[str, Any]) -> xr.Dataset:
        datasets = [ml_dataset.get_dataset(index) for ml_dataset in
                    self._ml_datasets]
        return self._combiner_function(datasets, **combiner_params)

    def close(self):
        for ml_dataset in self._ml_datasets:
            ml_dataset.close()


class MappedMultiLevelDataset(LazyMultiLevelDataset):
    def __init__(self,
                 ml_dataset: MultiLevelDataset,
                 mapper_function: Callable[
                     [xr.Dataset, Dict[str, Any]], xr.Dataset],
                 ds_id: str = None,
                 mapper_params: Dict[str, Any] = None):
        super().__init__(ds_id=ds_id,
                         parameters=mapper_params)
        self._ml_dataset = ml_dataset
        self._mapper_function = mapper_function

    def _get_num_levels_lazily(self) -> int:
        return self._ml_dataset.num_levels

    def _get_dataset_lazily(self, index: int,
                            mapper_params: Dict[str, Any]) -> xr.Dataset:
        return self._mapper_function(self._ml_dataset.get_dataset(index),
                                     **mapper_params)

    def close(self):
        self._ml_dataset.close()


class IdentityMultiLevelDataset(MappedMultiLevelDataset):
    def __init__(self, ml_dataset: MultiLevelDataset, ds_id: str = None):
        super().__init__(ml_dataset, lambda ds: ds, ds_id=ds_id)


class BaseMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset whose level datasets are
    created by down-sampling a base dataset.

    :param base_dataset: The base dataset for the level at index zero.
    :param grid_mapping: Optional grid mapping for *base_dataset*.
    :param num_levels: Optional number of levels.
    :param tile_grid: Optional tile grid.
    :param ds_id: Optional dataset identifier.
    :param agg_methods: Optional aggregation methods.
        May be given as string or as mapping from variable name pattern
        to aggregation method. Valid aggregation methods are
        None, "first", "min", "max", "mean", "median".
        If None, the default, "first" is used for integer variables
        and "mean" for floating point variables.
    """

    def __init__(self,
                 base_dataset: xr.Dataset,
                 grid_mapping: Optional[GridMapping] = None,
                 num_levels: Optional[int] = None,
                 agg_methods: AggMethods = 'first',
                 ds_id: Optional[str] = None):
        assert_instance(base_dataset, xr.Dataset, name='base_dataset')

        if grid_mapping is None:
            # TODO (forman): why not computing it lazily?
            grid_mapping = GridMapping.from_dataset(base_dataset,
                                                    tolerance=1e-4)

        assert_valid_agg_methods(agg_methods)
        self._agg_methods = agg_methods

        self._base_dataset = base_dataset
        super().__init__(grid_mapping=grid_mapping,
                         num_levels=num_levels,
                         ds_id=ds_id)

    def _get_num_levels_lazily(self) -> int:
        gm = self.grid_mapping
        return get_num_levels(gm.size, gm.tile_size)

    def _get_dataset_lazily(self,
                            index: int,
                            parameters: Dict[str, Any]) -> xr.Dataset:
        """
        Compute the dataset at level *index*: If *index* is zero, return
        the base image passed to constructor, otherwise down-sample the
        dataset for the level at given *index*.

        :param index: the level index
        :param parameters: currently unused
        :return: the dataset for the level at *index*.
        """
        assert_instance(index, int, name='index')
        if index < 0:
            index = self.num_levels + index
        assert_true(0 <= index < self.num_levels,
                    message='index out of range')

        if index == 0:
            level_dataset = self._base_dataset
        else:
            level_dataset = subsample_dataset(
                self._base_dataset,
                2 ** index,
                xy_dim_names=self.grid_mapping.xy_dim_names,
                agg_methods=self._agg_methods
            )

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
    A multi-level dataset whose level datasets are a computed from a Python
    script.

    The script can import other Python modules located in the same
    directory as *script_path*.
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

        # Allow scripts to import modules from
        # within directory
        script_parent = os.path.dirname(script_path)
        if script_parent not in sys.path:
            sys.path = [script_parent] + sys.path
            LOG.info(f'Python sys.path prepended by {script_parent}')

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
        LOG.info(f'Imported {callable_name}() from {script_path}')
        self._callable_name = callable_name
        self._callable_obj = callable_obj
        self._input_ml_dataset_ids = input_ml_dataset_ids
        self._input_ml_dataset_getter = input_ml_dataset_getter
        self._exception_type = exception_type

    def _get_num_levels_lazily(self) -> int:
        ds_0 = self._input_ml_dataset_getter(self._input_ml_dataset_ids[0])
        return ds_0.num_levels

    def _get_dataset_lazily(self, index: int, parameters: Dict[str, Any]) -> xr.Dataset:
        input_datasets = [self._input_ml_dataset_getter(ds_id).get_dataset(index)
                          for ds_id in self._input_ml_dataset_ids]
        try:
            with measure_time(tag=f"Computed in-memory dataset {self.ds_id!r} at level {index}"):
                computed_value = self._callable_obj(*input_datasets, **parameters)
        except Exception as e:
            raise self._exception_type(f"Failed to compute in-memory dataset {self.ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: {e}") from e
        if not isinstance(computed_value, xr.Dataset):
            raise self._exception_type(f"Failed to compute in-memory dataset {self.ds_id!r} at level {index} "
                                       f"from function {self._callable_name!r}: "
                                       f"expected an xarray.Dataset but got {type(computed_value)}")
        return computed_value


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


def open_ml_dataset_from_python_code(script_path: str,
                                     callable_name: str,
                                     input_ml_dataset_ids: Sequence[str] = None,
                                     input_ml_dataset_getter: Callable[[str], MultiLevelDataset] = None,
                                     input_parameters: Mapping[str, Any] = None,
                                     ds_id: str = None,
                                     exception_type: type = ValueError) -> MultiLevelDataset:
    with measure_time(tag=f"Opened memory dataset {script_path}"):
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
    with measure_time(tag=f"Added augmentation from {script_path}"):
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
