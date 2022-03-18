# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import fnmatch
import glob
import json
import os
import os.path
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Callable, Collection, Set
from typing import Sequence

from deprecated import deprecated
import fiona
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from xcube.constants import LOG
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.mldataset import augment_ml_dataset
from xcube.core.mldataset import open_ml_dataset_from_python_code
from xcube.core.normalize import decode_cube
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataStoreConfig
from xcube.core.store import DataStorePool
from xcube.core.store import DatasetDescriptor
from xcube.core.tile import get_var_cmap_params
from xcube.core.tile import get_var_valid_range
from xcube.util.cache import Cache
from xcube.util.cache import MemoryCacheStore
from xcube.util.cache import parse_mem_size
from xcube.util.cmaps import get_cmap
from xcube.util.perf import measure_time_cm
from xcube.util.tilegrid import TileGrid
from xcube.version import version
from xcube.webapi.defaults import DEFAULT_TRACE_PERF
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.errors import ServiceConfigError
from xcube.webapi.errors import ServiceError
from xcube.webapi.errors import ServiceResourceNotFoundError
from xcube.webapi.reqparams import RequestParams

COMPUTE_DATASET = 'compute_dataset'
COMPUTE_VARIABLES = 'compute_variables'

# We use tilde, because it is not a reserved URI characters
STORE_DS_ID_SEPARATOR = '~'
FS_TYPE_TO_PROTOCOL = {
    'local': 'file',
    'obs': 's3',
    'file': 'file',
    's3': 's3',
    'memory': 'memory'
}
NON_MEMORY_FILE_SYSTEMS = ['local', 'obs', 'file', 's3']

ALL_PLACES = "all"

Config = Dict[str, Any]
DatasetConfigDict = Dict[str, Any]

MultiLevelDatasetOpener = Callable[
    ["ServiceContext", DatasetConfigDict],
    MultiLevelDataset
]


# noinspection PyMethodMayBeStatic
class ServiceContext:

    def __init__(self,
                 prefix: str = None,
                 base_dir: str = None,
                 config: Config = None,
                 data_store_pool: DataStorePool = None,
                 trace_perf: bool = DEFAULT_TRACE_PERF,
                 tile_comp_mode: int = None,
                 tile_cache_capacity: int = None,
                 ml_dataset_openers: Dict[str, MultiLevelDatasetOpener] = None):
        self._prefix = normalize_prefix(prefix)
        self._base_dir = os.path.abspath(base_dir or '')
        self._config = config if config is not None else dict()
        self._config_mtime = 0.0
        self._place_group_cache = dict()
        self._feature_index = 0
        self._ml_dataset_openers = ml_dataset_openers
        self._tile_comp_mode = tile_comp_mode
        self._trace_perf = trace_perf
        self._lock = threading.RLock()
        # contains tuples of form (MultiLevelDataset, dataset_config)
        self._dataset_cache = dict()
        # cache for all dataset configs
        self._dataset_configs: Optional[List[DatasetConfigDict]] = None
        self._image_cache = dict()
        self._data_store_pool = data_store_pool or None
        if tile_cache_capacity:
            self._tile_cache = Cache(MemoryCacheStore(),
                                     capacity=tile_cache_capacity,
                                     threshold=0.75)
        else:
            self._tile_cache = None

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, config: Config):
        if self._config:
            with self._lock:
                # Close all datasets
                for ml_dataset, _ in self._dataset_cache.values():
                    # noinspection PyBroadException
                    try:
                        ml_dataset.close()
                    except Exception:
                        pass
                # Clear all caches
                if self._dataset_cache:
                    self._dataset_cache.clear()
                if self._image_cache:
                    self._image_cache.clear()
                if self._tile_cache:
                    self._tile_cache.clear()
                if self._place_group_cache:
                    self._place_group_cache.clear()
                if self._data_store_pool:
                    self._data_store_pool.remove_all_store_configs()
                self._dataset_configs = None
        self._config = config

    @property
    def config_mtime(self) -> float:
        return self._config_mtime

    @config_mtime.setter
    def config_mtime(self, value: float):
        self._config_mtime = value

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def tile_comp_mode(self) -> int:
        return self._tile_comp_mode

    @property
    def dataset_cache(self) -> Dict[str, Tuple[MultiLevelDataset, Dict[str, Any]]]:
        return self._dataset_cache

    @property
    def image_cache(self) -> Dict[str, Any]:
        return self._image_cache

    @property
    def tile_cache(self) -> Optional[Cache]:
        return self._tile_cache

    @property
    def trace_perf(self) -> bool:
        return self._trace_perf

    @property
    def measure_time(self):
        return measure_time_cm(disabled=not self.trace_perf, logger=LOG)

    @property
    def can_authenticate(self) -> bool:
        """
        Test whether the user can authenticate.
        Even if authentication service is configured, user authentication
        may still be optional. In this case the server will publish
        the resources configured to be free for everyone.
        """
        return bool(self.authentication.get('Domain'))

    @property
    def must_authenticate(self) -> bool:
        """
        Test whether the user must authenticate.
        """
        return self.can_authenticate \
               and self.authentication.get('IsRequired', False)

    @property
    def authentication(self) -> Dict[str, Any]:
        return self._config.get('Authentication', {})

    @property
    def access_control(self) -> Dict[str, Any]:
        return self._config.get('AccessControl', {})

    @property
    def required_scopes(self) -> List[str]:
        return self.access_control.get('RequiredScopes', [])

    def get_required_dataset_scopes(self,
                                    dataset_config: DatasetConfigDict) -> Set[str]:
        return self._get_required_scopes(dataset_config, 'read:dataset', 'Dataset',
                                         dataset_config['Identifier'])

    def get_required_variable_scopes(self,
                                     dataset_config: DatasetConfigDict,
                                     var_name: str) -> Set[str]:
        return self._get_required_scopes(dataset_config, 'read:variable', 'Variable',
                                         var_name)

    def _get_required_scopes(self,
                             dataset_config: DatasetConfigDict,
                             base_scope: str,
                             value_name: str,
                             value: str) -> Set[str]:
        required_global_scopes = set(self.required_scopes)
        required_dataset_scopes = set(
            dataset_config.get('AccessControl', {}).get('RequiredScopes', [])
        )
        if not required_global_scopes and not required_dataset_scopes:
            return required_global_scopes
        required_dataset_scopes = required_global_scopes \
            .union(required_dataset_scopes)
        base_scope_prefix = base_scope + ':'
        required_dataset_scopes = {
            scope
            for scope in required_dataset_scopes
            if scope == base_scope or scope.startswith(base_scope_prefix)
        }
        pattern_scope = base_scope_prefix + '{' + value_name + '}'
        if pattern_scope in required_dataset_scopes:
            # Replace "{base_scope}:{value_name}" by "{base_scope}:{value}"
            required_dataset_scopes.remove(pattern_scope)
            required_dataset_scopes.add(base_scope_prefix + value)
        return required_dataset_scopes

    def get_service_url(self, base_url, *path: str):
        # noinspection PyTypeChecker
        path_comp = '/'.join(path)
        if self._prefix:
            return base_url + self._prefix + '/' + path_comp
        else:
            return base_url + '/' + path_comp

    def get_ml_dataset(self, ds_id: str) -> MultiLevelDataset:
        ml_dataset, _ = self._get_dataset_entry(ds_id)
        return ml_dataset

    def set_ml_dataset(self, ml_dataset: MultiLevelDataset):
        self._set_dataset_entry((ml_dataset, dict(Identifier=ml_dataset.ds_id, Hidden=True)))

    def get_dataset(self, ds_id: str, expected_var_names: Collection[str] = None) -> xr.Dataset:
        ml_dataset, _ = self._get_dataset_entry(ds_id)
        dataset = ml_dataset.base_dataset
        if expected_var_names:
            for var_name in expected_var_names:
                if var_name not in dataset:
                    raise ServiceResourceNotFoundError(f'Variable "{var_name}" not found in dataset "{ds_id}"')
        return dataset

    def get_time_series_dataset(self, ds_id: str, var_name: str = None) -> xr.Dataset:
        dataset_config = self.get_dataset_config(ds_id)
        ts_ds_name = dataset_config.get('TimeSeriesDataset', ds_id)
        try:
            # Try to get more efficient, time-chunked dataset
            return self.get_dataset(ts_ds_name, expected_var_names=[var_name] if var_name else None)
        except ServiceResourceNotFoundError:
            # This happens, if the dataset pointed to by 'TimeSeriesDataset'
            # does not contain the variable given by var_name.
            return self.get_dataset(ds_id, expected_var_names=[var_name] if var_name else None)

    def get_variable_for_z(self, ds_id: str, var_name: str, z_index: int) -> xr.DataArray:
        ml_dataset = self.get_ml_dataset(ds_id)
        index = ml_dataset.num_levels - 1 - z_index
        if index < 0 or index >= ml_dataset.num_levels:
            raise ServiceResourceNotFoundError(f'Variable "{var_name}" has no z-index {z_index} in dataset "{ds_id}"')
        dataset = ml_dataset.get_dataset(index)
        if var_name not in dataset:
            raise ServiceResourceNotFoundError(f'Variable "{var_name}" not found in dataset "{ds_id}"')
        return dataset[var_name]

    def get_dataset_configs(self) -> List[DatasetConfigDict]:
        if self._dataset_configs is None:
            with self._lock:
                dataset_configs = self._config.get('Datasets', [])
                dataset_configs += \
                    self.get_dataset_configs_from_stores()
                self._dataset_configs = dataset_configs
                self._maybe_assign_store_instance_ids()
        return self._dataset_configs

    def _maybe_assign_store_instance_ids(self):
        assignable_dataset_configs = [dc for dc in self._dataset_configs
                                      if 'StoreInstanceId' not in dc
                                      and dc.get('FileSystem', 'file')
                                      in NON_MEMORY_FILE_SYSTEMS]
        # split into sublists according to file system and non-root store params
        config_lists = []
        for config in assignable_dataset_configs:
            store_params = self._get_other_store_params_than_root(config)
            file_system = config.get('FileSystem', 'file')
            appended = False
            for config_list in config_lists:
                if config_list[0] == file_system and \
                        config_list[1] == store_params:
                    config_list[2].append(config)
                    appended = True
                    break
            if not appended:
                config_lists.append((file_system, store_params, [config]))

        data_store_pool = self.get_data_store_pool()
        if not data_store_pool:
            data_store_pool = self._data_store_pool = DataStorePool()

        for file_system, store_params, config_list in config_lists:
            # Retrieve paths per configuration
            paths = [dc['Path'] for dc in config_list]
            list.sort(paths)
            # Determine common prefixes of paths (and call them roots)
            prefixes = _get_common_prefixes(paths)
            if len(prefixes) < 1:
                roots = ['']
            else:
                # perform further step to merge prefixes with same start
                prefixes = list(set(prefixes))
                prefixes.sort()
                roots = []
                root_candidate = prefixes[0]
                for root in prefixes[1:]:
                    common_root = os.path.commonprefix([root_candidate, root])
                    if _is_not_empty(common_root):
                        root_candidate = common_root
                    else:
                        roots.append(root_candidate)
                        root_candidate = root
                roots.append(root_candidate)
            for root in roots:
                # ensure root does not end with full or partial directory
                # or file name
                while not root.endswith("/") and not root.endswith("\\") and \
                        len(root) > 0:
                    root = root[:-1]
                if root.endswith("/") or root.endswith("\\"):
                    root = root[:-1]
                abs_root = root
                # For local file systems: Determine absolute root from base dir
                fs_protocol = FS_TYPE_TO_PROTOCOL.get(file_system, file_system)
                if fs_protocol == 'file' and not os.path.isabs(abs_root):
                    abs_root = os.path.join(self._base_dir, abs_root)
                    abs_root = os.path.normpath(abs_root)
                store_params_for_root = store_params.copy()
                store_params_for_root['root'] = abs_root
                # See if there already is a store with this configuration
                data_store_config = DataStoreConfig(
                    store_id=fs_protocol,
                    store_params=store_params_for_root)
                store_instance_id = data_store_pool.\
                    get_store_instance_id(data_store_config)
                if not store_instance_id:
                    # Create new store with new unique store instance id
                    counter = 1
                    while data_store_pool.has_store_instance(
                            f'{fs_protocol}_{counter}'):
                        counter += 1
                    store_instance_id = f'{fs_protocol}_{counter}'
                    data_store_pool.add_store_config(store_instance_id,
                                                     data_store_config)
                for config in config_list:
                    if config['Path'].startswith(root):
                        config['StoreInstanceId'] = store_instance_id
                        new_path = config['Path'][len(root):]
                        while new_path.startswith("/") or \
                                new_path.startswith("\\"):
                            new_path = new_path[1:]
                        config['Path'] = new_path

    def _get_other_store_params_than_root(self,
                                          dataset_config: DatasetConfigDict) \
            -> Dict:
        if FS_TYPE_TO_PROTOCOL.get(dataset_config.get('FileSystem',
                                                      'file')) != 's3':
            return {}
        storage_options=dict()
        if 'Anonymous' in dataset_config:
            storage_options['anon'] = dataset_config['Anonymous']
        client_kwargs = dict(
        )
        if 'Endpoint' in dataset_config:
            client_kwargs['endpoint_url'] = dataset_config['Endpoint']
        if 'Region' in dataset_config:
            client_kwargs['region_name'] = dataset_config['Region']
        storage_options['client_kwargs'] = client_kwargs
        store_params = dict(storage_options=storage_options)
        return store_params

    def get_dataset_configs_from_stores(self) \
            -> List[DatasetConfigDict]:

        data_store_pool = self.get_data_store_pool()
        if data_store_pool is None:
            return []

        all_dataset_configs: List[DatasetConfigDict] = []
        for store_instance_id in data_store_pool.store_instance_ids:
            LOG.info(f'scanning store {store_instance_id!r}')
            data_store_config = data_store_pool.get_store_config(
                store_instance_id
            )
            data_store = data_store_pool.get_store(store_instance_id)
            store_dataset_ids = data_store.get_data_ids(
                data_type=DATASET_TYPE
            )
            for store_dataset_id in store_dataset_ids:
                dataset_config_base = {}
                store_dataset_configs: List[DatasetConfigDict] \
                    = data_store_config.user_data
                if store_dataset_configs:
                    for store_dataset_config in store_dataset_configs:
                        dataset_id_pattern = store_dataset_config.get(
                            'Path', '*'
                        )
                        if fnmatch.fnmatch(store_dataset_id,
                                           dataset_id_pattern):
                            dataset_config_base = store_dataset_config
                            break
                        else:
                            dataset_config_base = None
                if dataset_config_base is not None:
                    LOG.debug(f'selected dataset {store_dataset_id!r}')
                    dataset_config = dict(
                        StoreInstanceId=store_instance_id,
                        **dataset_config_base
                    )
                    dataset_config['Path'] = store_dataset_id
                    dataset_config['Identifier'] = \
                        f'{store_instance_id}{STORE_DS_ID_SEPARATOR}' \
                        f'{store_dataset_id}'
                    all_dataset_configs.append(dataset_config)

        # Just for testing:
        debug_file = 'all_dataset_configs.json'
        with open(debug_file, 'w') as stream:
            json.dump(all_dataset_configs, stream)
            LOG.debug(f'wrote file {debug_file!r}')

        return all_dataset_configs

    def new_dataset_metadata(self,
                             store_instance_id: str,
                             dataset_id: str) -> Optional[DatasetDescriptor]:
        data_store = self._data_store_pool.get_store(store_instance_id)
        dataset_metadata = data_store.describe_data(
            dataset_id,
            data_type='dataset'
        )
        if dataset_metadata.crs is not None:
            crs = pyproj.CRS.from_string(dataset_metadata.crs)
            if not crs.is_geographic:
                LOG.warn(f'ignoring dataset {dataset_id!r} from'
                         f' store instance {store_instance_id!r}'
                         f' because it uses a non-geographic CRS')
                return None
        # noinspection PyTypeChecker
        return dataset_metadata

    def get_data_store_pool(self) -> Optional[DataStorePool]:
        data_store_configs = self._config.get('DataStores', [])
        if not data_store_configs or self._data_store_pool:
            return self._data_store_pool
        if not isinstance(data_store_configs, list):
            raise ServiceConfigError('DataStores must be a list')
        store_configs: Dict[str, DataStoreConfig] = {}
        for data_store_config_dict in data_store_configs:
            store_instance_id = data_store_config_dict.get('Identifier')
            store_id = data_store_config_dict.get('StoreId')
            store_params = data_store_config_dict.get('StoreParams', {})
            dataset_configs = data_store_config_dict.get('Datasets')
            store_config = DataStoreConfig(store_id,
                                           store_params=store_params,
                                           user_data=dataset_configs)
            store_configs[store_instance_id] = store_config
        self._data_store_pool = DataStorePool(store_configs)
        return self._data_store_pool

    def get_dataset_config(self, ds_id: str) -> Dict[str, Any]:
        dataset_configs = self.get_dataset_configs()
        dataset_config = self.find_dataset_config(
            dataset_configs, ds_id
        )
        if dataset_config is None:
            raise ServiceResourceNotFoundError(f'Dataset "{ds_id}" not found')
        return dataset_config

    def get_s3_bucket_mapping(self):
        s3_bucket_mapping = {}
        for dataset_config in self.get_dataset_configs():
            ds_id = dataset_config.get('Identifier')
            protocol = FS_TYPE_TO_PROTOCOL.get(dataset_config.get('FileSystem',
                                                                  'file'))
            if protocol == 'file':
                store_instance_id = dataset_config.get('StoreInstanceId')
                if store_instance_id:
                    data_store_pool = self.get_data_store_pool()
                    store_root = data_store_pool.get_store_config(
                        store_instance_id). \
                        store_params.get('root')
                    data_id = dataset_config.get('Path')
                    local_path = os.path.join(store_root, data_id)
                else:
                    local_path = self.get_config_path(dataset_config,
                                                      f'dataset configuration'
                                                      f' {ds_id!r}')
                local_path = os.path.normpath(local_path)
                if os.path.isdir(local_path):
                    s3_bucket_mapping[ds_id] = local_path
        return s3_bucket_mapping

    @deprecated(version='0.10.x',
                reason='do not use, wrong relationship')
    def get_tile_grid(self, ds_id: str) -> TileGrid:
        ml_dataset, _ = self._get_dataset_entry(ds_id)
        return ml_dataset.tile_grid

    def get_rgb_color_mapping(self,
                              ds_id: str,
                              norm_range: Tuple[float, float] = (0., 1.)) -> Tuple[List[Optional[str]],
                                                                                   List[Tuple[float, float]]]:
        var_names = [None, None, None]
        norm_ranges = [norm_range, norm_range, norm_range]
        color_mappings = self.get_color_mappings(ds_id)
        if color_mappings:
            rgb_mapping = color_mappings.get('rgb')
            if rgb_mapping:
                components = 'Red', 'Green', 'Blue'
                for i in range(3):
                    component = components[i]
                    component_config = rgb_mapping.get(component, {})
                    var_name = component_config.get('Variable')
                    norm_vmin, norm_vmax = component_config.get('ValueRange',
                                                                norm_range)
                    var_names[i] = var_name
                    norm_ranges[i] = norm_vmin, norm_vmax
        return var_names, norm_ranges

    def get_color_mapping(self, ds_id: str, var_name: str) -> Tuple[str, Tuple[float, float]]:
        cmap_name = None
        cmap_vmin, cmap_vmax = None, None
        color_mappings = self.get_color_mappings(ds_id)
        if color_mappings:
            color_mapping = color_mappings.get(var_name)
            if color_mapping:
                cmap_vmin, cmap_vmax = color_mapping.get('ValueRange', (None, None))
                if color_mapping.get('ColorFile') is not None:
                    cmap_name = color_mapping.get('ColorFile', cmap_name)
                else:
                    cmap_name = color_mapping.get('ColorBar', cmap_name)
                    cmap_name, _ = get_cmap(cmap_name)

        cmap_range = cmap_vmin, cmap_vmax
        if cmap_name is not None and None not in cmap_range:
            # noinspection PyTypeChecker
            return cmap_name, cmap_range

        ds = self.get_dataset(ds_id, expected_var_names=[var_name])
        var = ds[var_name]
        valid_range = get_var_valid_range(var)
        return get_var_cmap_params(var, cmap_name, cmap_range, valid_range)

    def get_style(self, ds_id: str):
        dataset_config = self.get_dataset_config(ds_id)
        style_name = dataset_config.get('Style', 'default')
        styles = self._config.get('Styles')
        if styles:
            for style in styles:
                if style_name == style['Identifier']:
                    return style
        return None

    def get_color_mappings(self, ds_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        style = self.get_style(ds_id)
        if style:
            return style.get('ColorMappings')
        return None

    def _get_dataset_entry(self, ds_id: str) -> Tuple[MultiLevelDataset, DatasetConfigDict]:
        if ds_id not in self._dataset_cache:
            with self._lock:
                self._set_dataset_entry(self._create_dataset_entry(ds_id))
        return self._dataset_cache[ds_id]

    def _set_dataset_entry(self, dataset_entry: Tuple[MultiLevelDataset, DatasetConfigDict]):
        ml_dataset, dataset_config = dataset_entry
        self._dataset_cache[ml_dataset.ds_id] = ml_dataset, dataset_config

    def _create_dataset_entry(self, ds_id: str) -> Tuple[MultiLevelDataset, Dict[str, Any]]:
        dataset_config = self.get_dataset_config(ds_id)
        ml_dataset = self._open_ml_dataset(dataset_config)
        return ml_dataset, dataset_config

    def _open_ml_dataset(self, dataset_config: DatasetConfigDict) \
            -> MultiLevelDataset:
        ds_id: str = dataset_config.get('Identifier')
        store_instance_id = dataset_config.get('StoreInstanceId')
        if store_instance_id:
            data_store_pool = self.get_data_store_pool()
            data_store = data_store_pool.get_store(store_instance_id)
            data_id = dataset_config.get('Path')
            open_params = dataset_config.get('StoreOpenParams') or {}
            # Inject chunk_cache_capacity into open parameters
            chunk_cache_capacity = self.get_dataset_chunk_cache_capacity(
                dataset_config
            )
            if chunk_cache_capacity \
                    and (data_id.endswith('.zarr')
                         or data_id.endswith('.levels')) \
                    and 'cache_size' not in open_params:
                open_params['cache_size'] = chunk_cache_capacity
            with self.measure_time(tag=f"opened dataset {ds_id!r}"
                                       f" from data store"
                                       f" {store_instance_id!r}"):
                dataset = data_store.open_data(data_id, **open_params)
            if isinstance(dataset, MultiLevelDataset):
                ml_dataset = dataset
            else:
                cube, _, _ = decode_cube(dataset,
                                         normalize=True,
                                         force_non_empty=True,
                                         force_geographic=True)
                ml_dataset = BaseMultiLevelDataset(cube, ds_id=ds_id)
        else:
            fs_type = dataset_config.get('FileSystem')
            if fs_type != 'memory':
                raise ServiceConfigError(f"Invalid FileSystem {fs_type!r}"
                                         f" in dataset configuration"
                                         f" {ds_id!r}")
            with self.measure_time(tag=f"opened dataset {ds_id!r}"
                                       f" from {fs_type!r}"):
                ml_dataset = _open_ml_dataset_from_python_code(self,
                                                               dataset_config)
        augmentation = dataset_config.get('Augmentation')
        if augmentation:
            script_path = self.get_config_path(
                augmentation,
                f"'Augmentation' of dataset configuration {ds_id}"
            )
            input_parameters = augmentation.get('InputParameters')
            callable_name = augmentation.get('Function', COMPUTE_VARIABLES)
            ml_dataset = augment_ml_dataset(
                ml_dataset,
                script_path,
                callable_name,
                self.get_ml_dataset,
                self.set_ml_dataset,
                input_parameters=input_parameters,
                exception_type=ServiceConfigError
            )
        return ml_dataset

    def get_legend_label(self, ds_id: str, var_name: str):
        dataset = self.get_dataset(ds_id)
        if var_name in dataset:
            ds = self.get_dataset(ds_id)
            units = ds[var_name].units
            return units
        raise ServiceResourceNotFoundError(f'Variable "{var_name}" not found in dataset "{ds_id}"')

    def get_dataset_place_groups(self, ds_id: str, base_url: str, load_features=False) -> List[Dict]:
        dataset_config = self.get_dataset_config(ds_id)

        place_group_id_prefix = f"DS-{ds_id}-"

        place_groups = []
        for k, v in self._place_group_cache.items():
            if k.startswith(place_group_id_prefix):
                place_groups.append(v)

        if place_groups:
            return place_groups

        place_groups = self._load_place_groups(dataset_config.get("PlaceGroups", []), base_url,
                                               is_global=False, load_features=load_features)
        for place_group in place_groups:
            self._place_group_cache[place_group_id_prefix + place_group["id"]] = place_group

        return place_groups

    def get_dataset_place_group(self, ds_id: str, place_group_id: str, base_url: str, load_features=False) -> Dict:
        place_groups = self.get_dataset_place_groups(ds_id, base_url, load_features=False)
        for place_group in place_groups:
            if place_group_id == place_group['id']:
                if load_features:
                    self._load_place_group_features(place_group)
                return place_group
        raise ServiceResourceNotFoundError(f'Place group "{place_group_id}" not found')

    def get_global_place_groups(self, base_url: str, load_features=False) -> List[Dict]:
        return self._load_place_groups(self._config.get("PlaceGroups", []),
                                       base_url,
                                       is_global=True,
                                       load_features=load_features)

    def get_global_place_group(self,
                               place_group_id: str,
                               base_url: str,
                               load_features: bool = False) -> Dict:
        place_group_config = self._get_place_group_config(place_group_id)
        return self._load_place_group(place_group_config, base_url, is_global=True, load_features=load_features)

    def _get_place_group_config(self, place_group_id: str) -> Dict:
        place_group_configs = self._config.get("PlaceGroups", [])
        for place_group_config in place_group_configs:
            if place_group_config['Identifier'] == place_group_id:
                return place_group_config
        raise ServiceResourceNotFoundError(f'Place group "{place_group_id}" not found')

    def _load_place_groups(self,
                           place_group_configs: Dict,
                           base_url: str,
                           is_global: bool = False,
                           load_features: bool = False) -> List[Dict]:
        place_groups = []
        for place_group_config in place_group_configs:
            place_group = self._load_place_group(place_group_config,
                                                 base_url,
                                                 is_global=is_global,
                                                 load_features=load_features)
            place_groups.append(place_group)
        return place_groups

    def _load_place_group(self,
                          place_group_config: Dict[str, Any],
                          base_url: str,
                          is_global: bool = False,
                          load_features: bool = False) -> Dict[str, Any]:
        place_group_id = place_group_config.get("PlaceGroupRef")
        if place_group_id:
            if is_global:
                raise ServiceConfigError("'PlaceGroupRef' cannot be used in a global place group")
            if len(place_group_config) > 1:
                raise ServiceConfigError("'PlaceGroupRef' if present, must be the only entry in a 'PlaceGroups' item")
            return self.get_global_place_group(place_group_id, base_url, load_features=load_features)

        place_group_id = place_group_config.get("Identifier")
        if not place_group_id:
            raise ServiceConfigError("Missing 'Identifier' entry in a 'PlaceGroups' item")

        if place_group_id in self._place_group_cache:
            place_group = self._place_group_cache[place_group_id]
        else:
            place_group_title = place_group_config.get("Title", place_group_id)
            place_path_wc = self.get_config_path(place_group_config, f"'PlaceGroups' item")
            source_paths = glob.glob(place_path_wc)
            source_encoding = place_group_config.get("CharacterEncoding", "utf-8")

            join = None
            place_join = place_group_config.get("Join")
            if isinstance(place_join, dict):
                join_path = self.get_config_path(place_join, "'Join' of a 'PlaceGroups' item")
                join_property = place_join.get("Property")
                if not join_property:
                    raise ServiceError("Missing 'Property' entry in 'Join' of a 'PlaceGroups' item")
                join_encoding = place_join.get("CharacterEncoding", "utf-8")
                join = dict(path=join_path, property=join_property, encoding=join_encoding)

            property_mapping = place_group_config.get("PropertyMapping")
            if property_mapping:
                property_mapping = dict(property_mapping)
                for key, value in property_mapping.items():
                    if isinstance(value, str) and '${base_url}' in value:
                        property_mapping[key] = value.replace('${base_url}', base_url)

            place_group = dict(type="FeatureCollection",
                               features=None,
                               id=place_group_id,
                               title=place_group_title,
                               propertyMapping=property_mapping,
                               sourcePaths=source_paths,
                               sourceEncoding=source_encoding,
                               join=join)

            sub_place_group_configs = place_group_config.get("Places")
            if sub_place_group_configs:
                raise ServiceConfigError("Invalid 'Places' entry in a 'PlaceGroups' item: not implemented yet")
            # sub_place_group_configs = place_group_config.get("Places")
            # if sub_place_group_configs:
            #     sub_place_groups = self._load_place_groups(sub_place_group_configs)
            #     place_group["placeGroups"] = sub_place_groups

            self._place_group_cache[place_group_id] = place_group

        if load_features:
            self._load_place_group_features(place_group)

        return place_group

    def _load_place_group_features(self, place_group: Dict[str, Any]) -> List[Dict[str, Any]]:
        features = place_group.get('features')
        if features is not None:
            return features
        source_files = place_group['sourcePaths']
        source_encoding = place_group['sourceEncoding']
        features = []
        for source_file in source_files:
            with fiona.open(source_file, encoding=source_encoding) as feature_collection:
                for feature in feature_collection:
                    self._remove_feature_id(feature)
                    feature["id"] = str(self._feature_index)
                    self._feature_index += 1
                    features.append(feature)

        join = place_group['join']
        if join:
            join_path = join['path']
            join_property = join['property']
            join_encoding = join['encoding']
            with fiona.open(join_path, encoding=join_encoding) as feature_collection:
                indexed_join_features = self._get_indexed_features(feature_collection, join_property)
            for feature in features:
                properties = feature.get('properties')
                if isinstance(properties, dict) and join_property in properties:
                    join_value = properties[join_property]
                    join_feature = indexed_join_features.get(join_value)
                    if join_feature:
                        join_properties = join_feature.get('properties')
                        if join_properties:
                            properties.update(join_properties)
                            feature['properties'] = properties

        place_group['features'] = features
        return features

    @classmethod
    def _get_indexed_features(cls, features: Sequence[Dict[str, Any]], property_name: str) -> Dict[Any, Any]:
        feature_index = {}
        for feature in features:
            properties = feature.get('properties')
            if properties and property_name in properties:
                property_value = properties[property_name]
                feature_index[property_value] = feature
        return feature_index

    @classmethod
    def _remove_feature_id(cls, feature: Dict):
        cls._remove_id(feature)

    @classmethod
    def _remove_id(cls, properties: Dict):
        if "id" in properties:
            del properties["id"]
        if "ID" in properties:
            del properties["ID"]

    def get_dataset_and_coord_variable(self, ds_name: str, dim_name: str):
        ds = self.get_dataset(ds_name)
        if dim_name not in ds.coords:
            raise ServiceResourceNotFoundError(f'Dimension {dim_name!r} has no coordinates in dataset {ds_name!r}')
        return ds, ds.coords[dim_name]

    @classmethod
    def get_var_indexers(cls,
                         ds_name: str,
                         var_name: str,
                         var: xr.DataArray,
                         dim_names: List[str],
                         params: RequestParams) -> Dict[str, Any]:
        var_indexers = dict()
        for dim_name in dim_names:
            if dim_name not in var.coords:
                raise ServiceBadRequestError(
                    f'dimension {dim_name!r} of variable {var_name!r} of dataset {ds_name!r} has no coordinates')
            coord_var = var.coords[dim_name]
            dim_value_str = params.get_query_argument(dim_name, None)
            try:
                if dim_value_str is None:
                    var_indexers[dim_name] = coord_var.values[0]
                elif dim_value_str == 'current':
                    var_indexers[dim_name] = coord_var.values[-1]
                elif np.issubdtype(coord_var.dtype, np.floating):
                    var_indexers[dim_name] = float(dim_value_str)
                elif np.issubdtype(coord_var.dtype, np.integer):
                    var_indexers[dim_name] = int(dim_value_str)
                elif np.issubdtype(coord_var.dtype, np.datetime64):
                    if '/' in dim_value_str:
                        date_str_1, date_str_2 = dim_value_str.split('/', maxsplit=1)
                        var_indexer_1 = pd.to_datetime(date_str_1)
                        var_indexer_2 = pd.to_datetime(date_str_2)
                        var_indexers[dim_name] = var_indexer_1 + (var_indexer_2 - var_indexer_1) / 2
                    else:
                        date_str = dim_value_str
                        var_indexers[dim_name] = pd.to_datetime(date_str)
                else:
                    raise ValueError(f'unable to convert value {dim_value_str!r} to {coord_var.dtype!r}')
            except ValueError as e:
                raise ServiceBadRequestError(
                    f'{dim_value_str!r} is not a valid value for dimension {dim_name!r} '
                    f'of variable {var_name!r} of dataset {ds_name!r}') from e
        return var_indexers

    @classmethod
    def find_dataset_config(cls,
                            dataset_configs: List[Dict[str, Any]],
                            ds_name: str) -> Optional[Dict[str, Any]]:
        # Note: can be optimized by dict/key lookup
        return next((dsd for dsd in dataset_configs if dsd['Identifier'] == ds_name), None)

    def get_config_path(self,
                        config: Dict[str, Any],
                        config_name: str,
                        path_entry_name: str = 'Path',
                        is_url: bool = False) -> str:
        path = config.get(path_entry_name)
        if not path:
            raise ServiceError(f"Missing entry {path_entry_name!r} in {config_name}")
        if not is_url and not os.path.isabs(path):
            path = os.path.join(self._base_dir, path)
        return path

    def get_dataset_chunk_cache_capacity(self, dataset_config: DatasetConfigDict) -> Optional[int]:
        cache_size = self.get_chunk_cache_capacity(dataset_config, 'ChunkCacheSize')
        if cache_size is None:
            cache_size = self.get_chunk_cache_capacity(self.config, 'DatasetChunkCacheSize')
        return cache_size

    @classmethod
    def get_chunk_cache_capacity(cls, config: Dict[str, Any], cache_size_key: str) -> Optional[int]:
        cache_size = config.get(cache_size_key, None)
        if not cache_size:
            return None
        elif isinstance(cache_size, str):
            try:
                cache_size = parse_mem_size(cache_size)
            except ValueError:
                raise ServiceConfigError(f'Invalid {cache_size_key}')
        elif not isinstance(cache_size, int) or cache_size < 0:
            raise ServiceConfigError(f'Invalid {cache_size_key}')
        return cache_size


def normalize_prefix(prefix: Optional[str]) -> str:
    if not prefix:
        return ''

    prefix = prefix.replace('${name}', 'xcube')
    prefix = prefix.replace('${version}', version)
    prefix = prefix.replace('//', '/').replace('//', '/')

    if prefix == '/':
        return ''

    if not prefix.startswith('/'):
        prefix = '/' + prefix

    if prefix.endswith('/'):
        prefix = prefix[0:-1]

    return prefix


def _open_ml_dataset_from_python_code(ctx: ServiceContext,
                                      dataset_config: DatasetConfigDict) -> MultiLevelDataset:
    ds_id = dataset_config.get('Identifier')
    path = ctx.get_config_path(dataset_config,
                               f"dataset configuration {ds_id}")
    callable_name = dataset_config.get('Function', COMPUTE_DATASET)
    input_dataset_ids = dataset_config.get('InputDatasets', [])
    input_parameters = dataset_config.get('InputParameters', {})
    chunk_cache_capacity = ctx.get_dataset_chunk_cache_capacity(dataset_config)
    if chunk_cache_capacity:
        warnings.warn('chunk cache size is not effective for datasets computed from scripts')
    for input_dataset_id in input_dataset_ids:
        if not ctx.get_dataset_config(input_dataset_id):
            raise ServiceConfigError(f"Invalid dataset configuration {ds_id!r}: "
                                     f"Input dataset {input_dataset_id!r} of callable {callable_name!r} "
                                     f"must reference another dataset")
    return open_ml_dataset_from_python_code(path,
                                            callable_name=callable_name,
                                            input_ml_dataset_ids=input_dataset_ids,
                                            input_ml_dataset_getter=ctx.get_ml_dataset,
                                            input_parameters=input_parameters,
                                            ds_id=ds_id,
                                            exception_type=ServiceConfigError)


def _is_not_empty(prefix):
    return prefix != '' and prefix != '/' and prefix != '\\'


def _get_common_prefixes(p):
    # Recursively examine a list of paths for common prefixes:
    # If no common prefix is found, split the list in half and
    # examine each half separately
    prefix = os.path.commonprefix(p)
    if _is_not_empty(prefix) or len(p) == 1:
        return [prefix]
    else:
        return _get_common_prefixes(p[:int(len(p) / 2)]) + \
               _get_common_prefixes(p[int(len(p) / 2):])


_MULTI_LEVEL_DATASET_OPENERS = {
    "memory": _open_ml_dataset_from_python_code,
}