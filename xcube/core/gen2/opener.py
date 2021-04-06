# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import warnings
from typing import Sequence, Tuple

import xarray as xr

from xcube.core.normalize import normalize_dataset
from xcube.core.select import select_subset
from xcube.core.store import DataStorePool
from xcube.core.store import TYPE_SPECIFIER_CUBE
from xcube.core.store import TYPE_SPECIFIER_DATASET
from xcube.core.store import get_data_store_instance
from xcube.core.store import new_data_opener
from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .config import CubeConfig
from .config import InputConfig
from .error import CubeGeneratorError

SUBSET_PARAMETER_NAMES = ('variable_names', 'bbox', 'time_range')
CHUNKS_PARAMETER_NAME = 'chunks'


class CubesOpener:

    def __init__(self,
                 input_configs: Sequence[InputConfig],
                 cube_config: CubeConfig,
                 store_pool: DataStorePool = None):
        assert_condition(len(input_configs) > 0,
                         'At least one input must be given')
        assert_instance(cube_config, CubeConfig, 'cube_config')
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, 'store_pool')
        self._input_configs = input_configs
        self._cube_config = cube_config
        self._store_pool = store_pool

    def open_cubes(self) -> Sequence[xr.Dataset]:
        cubes = []
        with observe_progress('Opening input dataset(s)', len(self._input_configs)) as progress:
            for input_config in self._input_configs:
                cubes.append(self._open_cube(input_config))
                progress.worked(1)
        return cubes

    def _open_cube(self, input_config: InputConfig) -> xr.Dataset:
        cube_params = self._cube_config.to_dict()
        opener_id = input_config.opener_id
        store_params = input_config.store_params or {}
        open_params = input_config.open_params or {}
        normalisation_required = False
        if input_config.store_id:
            store_instance = get_data_store_instance(input_config.store_id,
                                                     store_params=store_params,
                                                     store_pool=self._store_pool)
            store = store_instance.store
            if opener_id is None:
                opener_id, normalisation_required = self._get_opener_id(input_config, store)
            opener = store
            open_params.update(opener_id=opener_id, **open_params)
        else:
            opener = new_data_opener(opener_id)
            open_params.update(**store_params, **open_params)
        open_params_schema = opener.get_open_data_params_schema(input_config.data_id)
        cube_open_params = {k: v for k, v in cube_params.items() if k in open_params_schema.properties}
        unsupported_cube_params = {k for k in cube_params.keys() if k not in open_params_schema.properties}
        unsupported_cube_subset_params = {}
        if unsupported_cube_params:
            for k in unsupported_cube_params:
                if k in SUBSET_PARAMETER_NAMES:
                    unsupported_cube_subset_params[k] = cube_params[k]
                    warnings.warn(f'cube_config parameter not supported by data store or opener:'
                                  f' manually applying: {k} = {cube_params[k]!r}')
                elif k != CHUNKS_PARAMETER_NAME:
                    warnings.warn(f'cube_config parameter not supported by data store or opener:'
                                  f' ignoring {k} = {cube_params[k]!r}')
        cube = opener.open_data(input_config.data_id, **open_params, **cube_open_params)
        if normalisation_required:
            cube = normalize_dataset(cube)
        if unsupported_cube_subset_params:
            # Try creating subsets given the cube subset parameters not supported
            # by store in use. Note that we expect some trouble when using bbox
            # without properly recognising spatial_res or crs. Especially for
            # multiple inputs with different source spatial_res the bboxes are no
            # longer aligned. The new resampling module will need to account
            # for this.
            cube = select_subset(cube,
                                 var_names=unsupported_cube_subset_params.get('variable_names'),
                                 bbox=unsupported_cube_subset_params.get('bbox'),
                                 time_range=unsupported_cube_subset_params.get('time_range'))
        return cube

    @classmethod
    def _get_opener_id(cls, input_config, store) -> Tuple[str, bool]:
        normalisation_required = False
        opener_ids = None
        type_specifiers = store.get_type_specifiers_for_data(input_config.data_id)
        for type_specifier in type_specifiers:
            if TYPE_SPECIFIER_CUBE.is_satisfied_by(type_specifier):
                opener_ids = \
                    store.get_data_opener_ids(data_id=input_config.data_id,
                                              type_specifier=type_specifier)
                break
        if not opener_ids:
            for type_specifier in type_specifiers:
                if TYPE_SPECIFIER_DATASET.is_satisfied_by(type_specifier):
                    opener_ids = \
                        store.get_data_opener_ids(data_id=input_config.data_id,
                                                  type_specifier=type_specifier)
                    normalisation_required = True
                    break
        if not opener_ids:
            raise CubeGeneratorError(f'Data store "{input_config.store_id}" '
                                     f'does not support datasets')
        opener_id = opener_ids[0]
        return opener_id, normalisation_required
