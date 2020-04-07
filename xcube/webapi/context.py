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

import glob
import os
import os.path
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Callable, Collection, Set
from typing import Sequence

import fiona
import numpy as np
import pandas as pd
import xarray as xr

from xcube.constants import FORMAT_NAME_ZARR, LOG
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.mldataset import augment_ml_dataset
from xcube.core.mldataset import open_ml_dataset_from_local_fs
from xcube.core.mldataset import open_ml_dataset_from_object_storage
from xcube.core.mldataset import open_ml_dataset_from_python_code
from xcube.core.tile import get_var_cmap_params
from xcube.core.tile import get_var_valid_range
from xcube.util.cache import MemoryCacheStore, Cache, parse_mem_size
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

ALL_PLACES = "all"

Config = Dict[str, Any]
DatasetDescriptor = Dict[str, Any]

MultiLevelDatasetOpener = Callable[["ServiceContext", DatasetDescriptor], MultiLevelDataset]


# noinspection PyMethodMayBeStatic
class ServiceContext:

    def __init__(self,
                 prefix: str = None,
                 base_dir: str = None,
                 config: Config = None,
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
        self._dataset_cache = dict()  # contains tuples of form (MultiLevelDataset, ds_descriptor)
        self._image_cache = dict()
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
    def access_control(self) -> Dict[str, Any]:
        return dict(self._config.get('AccessControl', {}))

    @property
    def required_scopes(self) -> List[str]:
        return self.access_control.get('RequiredScopes', [])

    def get_required_dataset_scopes(self,
                                    dataset_descriptor: DatasetDescriptor) -> Set[str]:
        return self._get_required_scopes(dataset_descriptor, 'read:dataset', 'Dataset',
                                         dataset_descriptor['Identifier'])

    def get_required_variable_scopes(self,
                                     dataset_descriptor: DatasetDescriptor,
                                     var_name: str) -> Set[str]:
        return self._get_required_scopes(dataset_descriptor, 'read:variable', 'Variable',
                                         var_name)

    def _get_required_scopes(self,
                             dataset_descriptor: DatasetDescriptor,
                             base_scope: str,
                             value_name: str,
                             value: str) -> Set[str]:
        base_scope_prefix = base_scope + ':'
        pattern_scope = base_scope_prefix + '{' + value_name + '}'
        dataset_access_control = dataset_descriptor.get('AccessControl', {})
        dataset_required_scopes = dataset_access_control.get('RequiredScopes', [])
        dataset_required_scopes = set(self.required_scopes + dataset_required_scopes)
        dataset_required_scopes = {scope for scope in dataset_required_scopes
                                   if scope == base_scope or scope.startswith(base_scope_prefix)}
        if pattern_scope in dataset_required_scopes:
            dataset_required_scopes.remove(pattern_scope)
            dataset_required_scopes.add(base_scope_prefix + value)
        return dataset_required_scopes

    def get_service_url(self, base_url, *path: str):
        if self._prefix:
            return base_url + '/' + self._prefix + '/' + '/'.join(path)
        else:
            return base_url + '/' + '/'.join(path)

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
        descriptor = self.get_dataset_descriptor(ds_id)
        ts_ds_name = descriptor.get('TimeSeriesDataset', ds_id)
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

    def get_dataset_descriptors(self):
        dataset_descriptors = self._config.get('Datasets')
        if not dataset_descriptors:
            raise ServiceConfigError(f"No datasets configured")
        return dataset_descriptors

    def get_dataset_descriptor(self, ds_id: str) -> Dict[str, Any]:
        dataset_descriptors = self.get_dataset_descriptors()
        if not dataset_descriptors:
            raise ServiceConfigError(f"No datasets configured")
        dataset_descriptor = self.find_dataset_descriptor(dataset_descriptors, ds_id)
        if dataset_descriptor is None:
            raise ServiceResourceNotFoundError(f'Dataset "{ds_id}" not found')
        return dataset_descriptor

    def get_s3_bucket_mapping(self):
        s3_bucket_mapping = {}
        for descriptor in self.get_dataset_descriptors():
            ds_id = descriptor.get('Identifier')
            file_system = descriptor.get('FileSystem', 'local')
            if file_system == 'local':
                local_path = self.get_descriptor_path(descriptor, f'dataset descriptor {ds_id}')
                local_path = os.path.normpath(local_path)
                if os.path.isdir(local_path):
                    s3_bucket_mapping[ds_id] = local_path
        return s3_bucket_mapping

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
                    c = components[i]
                    c_descriptor = rgb_mapping.get(c, {})
                    var_name = c_descriptor.get('Variable')
                    norm_vmin, norm_vmax = c_descriptor.get('ValueRange', norm_range)
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
        dataset_descriptor = self.get_dataset_descriptor(ds_id)
        style_name = dataset_descriptor.get('Style', 'default')
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

    def _get_dataset_entry(self, ds_id: str) -> Tuple[MultiLevelDataset, DatasetDescriptor]:
        if ds_id not in self._dataset_cache:
            with self._lock:
                self._set_dataset_entry(self._create_dataset_entry(ds_id))
        return self._dataset_cache[ds_id]

    def _set_dataset_entry(self, dataset_entry: Tuple[MultiLevelDataset, DatasetDescriptor]):
        ml_dataset, dataset_descriptor = dataset_entry
        self._dataset_cache[ml_dataset.ds_id] = ml_dataset, dataset_descriptor

    def _create_dataset_entry(self, ds_id: str) -> Tuple[MultiLevelDataset, Dict[str, Any]]:
        dataset_descriptor = self.get_dataset_descriptor(ds_id)
        ml_dataset = self._open_ml_dataset(dataset_descriptor)
        return ml_dataset, dataset_descriptor

    def _open_ml_dataset(self, dataset_descriptor: DatasetDescriptor) -> MultiLevelDataset:
        ds_id = dataset_descriptor.get('Identifier')
        fs_type = dataset_descriptor.get('FileSystem', 'local')
        if self._ml_dataset_openers and fs_type in self._ml_dataset_openers:
            ml_dataset_opener = self._ml_dataset_openers[fs_type]
        elif fs_type in _DEFAULT_MULTI_LEVEL_DATASET_OPENERS:
            ml_dataset_opener = _DEFAULT_MULTI_LEVEL_DATASET_OPENERS[fs_type]
        else:
            raise ServiceConfigError(f"Invalid fs={fs_type!r} in dataset descriptor {ds_id!r}")
        ml_dataset = ml_dataset_opener(self, dataset_descriptor)
        augmentation = dataset_descriptor.get('Augmentation')
        if augmentation:
            script_path = self.get_descriptor_path(augmentation,
                                                   f"'Augmentation' of dataset descriptor {ds_id}")
            input_parameters = augmentation.get('InputParameters')
            callable_name = augmentation.get('Function', COMPUTE_VARIABLES)
            ml_dataset = augment_ml_dataset(ml_dataset,
                                            script_path,
                                            callable_name,
                                            self.get_ml_dataset,
                                            self.set_ml_dataset,
                                            input_parameters=input_parameters,
                                            exception_type=ServiceConfigError)
        return ml_dataset

    def get_legend_label(self, ds_id: str, var_name: str):
        dataset = self.get_dataset(ds_id)
        if var_name in dataset:
            ds = self.get_dataset(ds_id)
            units = ds[var_name].units
            return units
        raise ServiceResourceNotFoundError(f'Variable "{var_name}" not found in dataset "{ds_id}"')

    def get_dataset_place_groups(self, ds_id: str, base_url: str, load_features=False) -> List[Dict]:
        dataset_descriptor = self.get_dataset_descriptor(ds_id)

        place_group_id_prefix = f"DS-{ds_id}-"

        place_groups = []
        for k, v in self._place_group_cache.items():
            if k.startswith(place_group_id_prefix):
                place_groups.append(v)

        if place_groups:
            return place_groups

        place_groups = self._load_place_groups(dataset_descriptor.get("PlaceGroups", []), base_url,
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
        place_group_descriptor = self._get_place_group_descriptor(place_group_id)
        return self._load_place_group(place_group_descriptor, base_url, is_global=True, load_features=load_features)

    def _get_place_group_descriptor(self, place_group_id: str) -> Dict:
        place_group_descriptors = self._config.get("PlaceGroups", [])
        for place_group_descriptor in place_group_descriptors:
            if place_group_descriptor['Identifier'] == place_group_id:
                return place_group_descriptor
        raise ServiceResourceNotFoundError(f'Place group "{place_group_id}" not found')

    def _load_place_groups(self,
                           place_group_descriptors: Dict,
                           base_url: str,
                           is_global: bool = False,
                           load_features: bool = False) -> List[Dict]:
        place_groups = []
        for place_group_descriptor in place_group_descriptors:
            place_group = self._load_place_group(place_group_descriptor,
                                                 base_url,
                                                 is_global=is_global,
                                                 load_features=load_features)
            place_groups.append(place_group)
        return place_groups

    def _load_place_group(self,
                          place_group_descriptor: Dict[str, Any],
                          base_url: str,
                          is_global: bool = False,
                          load_features: bool = False) -> Dict[str, Any]:
        place_group_id = place_group_descriptor.get("PlaceGroupRef")
        if place_group_id:
            if is_global:
                raise ServiceConfigError("'PlaceGroupRef' cannot be used in a global place group")
            if len(place_group_descriptor) > 1:
                raise ServiceConfigError("'PlaceGroupRef' if present, must be the only entry in a 'PlaceGroups' item")
            return self.get_global_place_group(place_group_id, base_url, load_features=load_features)

        place_group_id = place_group_descriptor.get("Identifier")
        if not place_group_id:
            raise ServiceConfigError("Missing 'Identifier' entry in a 'PlaceGroups' item")

        if place_group_id in self._place_group_cache:
            place_group = self._place_group_cache[place_group_id]
        else:
            place_group_title = place_group_descriptor.get("Title", place_group_id)
            place_path_wc = self.get_descriptor_path(place_group_descriptor, f"'PlaceGroups' item")
            source_paths = glob.glob(place_path_wc)
            source_encoding = place_group_descriptor.get("CharacterEncoding", "utf-8")

            join = None
            place_join = place_group_descriptor.get("Join")
            if isinstance(place_join, dict):
                join_path = self.get_descriptor_path(place_join, "'Join' of a 'PlaceGroups' item")
                join_property = place_join.get("Property")
                if not join_property:
                    raise ServiceError("Missing 'Property' entry in 'Join' of a 'PlaceGroups' item")
                join_encoding = place_join.get("CharacterEncoding", "utf-8")
                join = dict(path=join_path, property=join_property, encoding=join_encoding)

            property_mapping = place_group_descriptor.get("PropertyMapping")
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

            sub_place_group_configs = place_group_descriptor.get("Places")
            if sub_place_group_configs:
                raise ServiceConfigError("Invalid 'Places' entry in a 'PlaceGroups' item: not implemented yet")
            # sub_place_group_descriptors = place_group_config.get("Places")
            # if sub_place_group_descriptors:
            #     sub_place_groups = self._load_place_groups(sub_place_group_descriptors)
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
    def find_dataset_descriptor(cls,
                                dataset_descriptors: List[Dict[str, Any]],
                                ds_name: str) -> Optional[Dict[str, Any]]:
        # Note: can be optimized by dict/key lookup
        return next((dsd for dsd in dataset_descriptors if dsd['Identifier'] == ds_name), None)

    def get_descriptor_path(self,
                            descriptor: Dict[str, Any],
                            descriptor_name: str,
                            path_entry_name: str = 'Path',
                            is_url: bool = False) -> str:
        path = descriptor.get(path_entry_name)
        if not path:
            raise ServiceError(f"Missing entry {path_entry_name!r} in {descriptor_name}")
        if not is_url and not os.path.isabs(path):
            path = os.path.join(self._base_dir, path)
        return path

    def get_dataset_chunk_cache_capacity(self, dataset_descriptor: DatasetDescriptor) -> Optional[int]:
        cache_size = self.get_chunk_cache_capacity(dataset_descriptor, 'ChunkCacheSize')
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


def normalize_prefix(prefix: Optional[str]):
    if not prefix:
        return ''

    prefix = prefix.replace('${version}', version).replace('${name}', 'xcube')
    if not prefix.startswith('/'):
        return '/' + prefix

    return prefix


# noinspection PyUnusedLocal
def _open_ml_dataset_from_object_storage(ctx: ServiceContext,
                                         dataset_descriptor: DatasetDescriptor) -> MultiLevelDataset:
    ds_id = dataset_descriptor.get('Identifier')
    path = ctx.get_descriptor_path(dataset_descriptor, f"dataset descriptor {ds_id}", is_url=True)
    data_format = dataset_descriptor.get('Format', FORMAT_NAME_ZARR)
    client_kwargs = dict()
    endpoint_url = None
    if 'Endpoint' in dataset_descriptor:
        client_kwargs['endpoint_url'] = dataset_descriptor['Endpoint']
    region_name = None
    if 'Region' in dataset_descriptor:
        client_kwargs['region_name'] = dataset_descriptor['Region']
    chunk_cache_capacity = ctx.get_dataset_chunk_cache_capacity(dataset_descriptor)
    return open_ml_dataset_from_object_storage(path,
                                               data_format=data_format,
                                               ds_id=ds_id,
                                               exception_type=ServiceConfigError,
                                               client_kwargs=client_kwargs,
                                               chunk_cache_capacity=chunk_cache_capacity)


def _open_ml_dataset_from_local_fs(ctx: ServiceContext,
                                   dataset_descriptor: DatasetDescriptor) -> MultiLevelDataset:
    ds_id = dataset_descriptor.get('Identifier')
    path = ctx.get_descriptor_path(dataset_descriptor, f"dataset descriptor {ds_id}")
    data_format = dataset_descriptor.get('Format')
    chunk_cache_capacity = ctx.get_dataset_chunk_cache_capacity(dataset_descriptor)
    if chunk_cache_capacity:
        warnings.warn('chunk cache size is not effective for datasets stored in local file systems')
    return open_ml_dataset_from_local_fs(path,
                                         data_format=data_format,
                                         ds_id=ds_id,
                                         exception_type=ServiceConfigError)


def _open_ml_dataset_from_python_code(ctx: ServiceContext,
                                      dataset_descriptor: DatasetDescriptor) -> MultiLevelDataset:
    ds_id = dataset_descriptor.get('Identifier')
    path = ctx.get_descriptor_path(dataset_descriptor, f"dataset descriptor {ds_id}")
    callable_name = dataset_descriptor.get('Function', COMPUTE_DATASET)
    input_dataset_ids = dataset_descriptor.get('InputDatasets', [])
    input_parameters = dataset_descriptor.get('InputParameters', {})
    chunk_cache_capacity = ctx.get_dataset_chunk_cache_capacity(dataset_descriptor)
    if chunk_cache_capacity:
        warnings.warn('chunk cache size is not effective for datasets computed from scripts')
    for input_dataset_id in input_dataset_ids:
        if not ctx.get_dataset_descriptor(input_dataset_id):
            raise ServiceConfigError(f"Invalid dataset descriptor {ds_id!r}: "
                                     f"Input dataset {input_dataset_id!r} of callable {callable_name!r} "
                                     f"must reference another dataset")
    return open_ml_dataset_from_python_code(path,
                                            callable_name=callable_name,
                                            input_ml_dataset_ids=input_dataset_ids,
                                            input_ml_dataset_getter=ctx.get_ml_dataset,
                                            input_parameters=input_parameters,
                                            ds_id=ds_id,
                                            exception_type=ServiceConfigError)


_DEFAULT_MULTI_LEVEL_DATASET_OPENERS = {
    "obs": _open_ml_dataset_from_object_storage,
    "local": _open_ml_dataset_from_local_fs,
    "memory": _open_ml_dataset_from_python_code,
}
