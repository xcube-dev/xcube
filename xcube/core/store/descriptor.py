# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from typing import Tuple, Sequence, Mapping, Optional, Dict, Any, Union

import numpy as np
import geopandas as gpd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store.typespecifier import get_type_specifier
from xcube.core.store.typespecifier import TypeSpecifier
from xcube.core.store.typespecifier import TYPE_SPECIFIER_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.ipython import register_json_formatter
from xcube.util.jsonschema import JsonObjectSchema

# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.schema.CubeSchema class
#   xcube.webapi.context.DatasetDescriptor type
#   responses of xcube.webapi.controllers.catalogue
# TODO: write tests
# TODO: validate params


def new_data_descriptor(data_id: str, data: Any) -> 'DataDescriptor':
    if isinstance(data, xr.Dataset):
        # TODO: implement me: data -> DatasetDescriptor
        return DatasetDescriptor(data_id=data_id, type_specifier=get_type_specifier(data))
    elif isinstance(data, MultiLevelDataset):
        # TODO: implement me: data -> MultiLevelDatasetDescriptor
        return MultiLevelDatasetDescriptor(data_id=data_id, num_levels=5)
    elif isinstance(data, gpd.GeoDataFrame):
        # TODO: implement me: data -> GeoDataFrameDescriptor
        return GeoDataFrameDescriptor(data_id=data_id, num_levels=5)
    raise NotImplementedError()


class DataDescriptor:
    """
    A generic descriptor for any data.
    Also serves as a base class for more specific data descriptors.
    """

    def __init__(self,
                 data_id: str,
                 type_specifier: Union[str, TypeSpecifier],
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 spatial_res: float = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 open_params_schema: JsonObjectSchema = None):
        assert_given(data_id, 'data_id')
        assert_given(type_specifier, 'type_specifier')
        self.data_id = data_id
        self.type_specifier = TypeSpecifier.normalize(type_specifier)
        self.crs = crs
        self.bbox = tuple(bbox) if bbox else None
        self.spatial_res = spatial_res
        self.time_range = tuple(time_range) if time_range else None
        self.time_period = time_period
        self.open_params_schema = open_params_schema

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict(type_specifier=str(self.type_specifier))
        _copy_none_null_props(self, d, ['data_id', 'crs', 'bbox', 'spatial_res', 'time_range', 'time_period'])

        if self.open_params_schema is not None:
            d['open_params_schema'] = self.open_params_schema.to_dict()
        return d


class DatasetDescriptor(DataDescriptor):
    """
    A descriptor for a gridded, N-dimensional dataset represented by xarray.Dataset.
    Comprises a description of the data variables contained in the dataset.
    """

    def __init__(self,
                 data_id: str,
                 type_specifier: Union[str, TypeSpecifier] = TYPE_SPECIFIER_DATASET,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 spatial_res: float = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 dims: Mapping[str, int] = None,
                 data_vars: Sequence['VariableDescriptor'] = None,
                 attrs: Mapping[str, any] = None,
                 open_params_schema: JsonObjectSchema = None):
        self._assert_valid_type_specifier(type_specifier)
        super().__init__(data_id=data_id,
                         type_specifier=type_specifier,
                         crs=crs,
                         bbox=bbox,
                         spatial_res=spatial_res,
                         time_range=time_range,
                         time_period=time_period,
                         open_params_schema=open_params_schema)
        self.dims = dict(dims) if dims else None
        self.data_vars = list(data_vars) if data_vars else None
        self.attrs = _convert_nans_to_none(dict(attrs)) if attrs else None

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_DATASET

    @classmethod
    def _assert_valid_type_specifier(cls, type_specifier: str):
        base_type_specifier = cls._get_base_type_specifier()
        if not base_type_specifier.is_satisfied_by(type_specifier):
            raise ValueError('type_specifier must satisfy'
                             f' type specifier "{base_type_specifier}", was "{type_specifier}"')

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('data_id', d)
        data_vars = []
        data_vars_as_dicts = d.get('data_vars', [])
        for data_var_as_dict in data_vars_as_dicts:
            data_vars.append(VariableDescriptor.from_dict(data_var_as_dict))
        return DatasetDescriptor(data_id=d['data_id'],
                                 type_specifier=d.get('type_specifier', TYPE_SPECIFIER_DATASET),
                                 crs=d.get('crs', None),
                                 bbox=d.get('bbox', None),
                                 spatial_res=d.get('spatial_res', None),
                                 time_range=d.get('time_range', None),
                                 time_period=d.get('time_period', None),
                                 dims=d.get('dims', None),
                                 data_vars=data_vars,
                                 attrs=d.get('attrs', None),
                                 open_params_schema=d.get('open_params_schema', None))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        if self.data_vars is not None:
            d['data_vars'] = [vd.to_dict() for vd in self.data_vars]
        _copy_none_null_props(self, d, ['dims', 'attrs'])
        return d


class VariableDescriptor:
    """
    A descriptor for dataset variable represented by xarray.DataArray instances.
    They are part of dataset descriptor for an gridded, N-dimensional dataset represented by xarray.Dataset.
    """

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 attrs: Mapping[str, any] = None):
        assert_given(name, 'name')
        assert_given(dtype, 'dtype')
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.attrs = _convert_nans_to_none(dict(attrs)) if attrs is not None else None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('name', d)
        assert_in('dtype', d)
        assert_in('dims', d)
        return VariableDescriptor(d['name'], d['dtype'], d['dims'], d.get('attrs', None))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict(name=self.name, dtype=self.dtype)
        _copy_none_null_props(self, d, ['dims', 'ndim', 'attrs'])
        return d


class MultiLevelDatasetDescriptor(DatasetDescriptor):
    """
    A descriptor for a gridded, N-dimensional, multi-level, multi-resolution dataset represented by
    xcube.core.mldataset.MultiLevelDataset.
    """

    def __init__(self,
                 data_id: str,
                 num_levels: int,
                 type_specifier: Union[str, TypeSpecifier] = TYPE_SPECIFIER_MULTILEVEL_DATASET,
                 **kwargs):
        assert_given(data_id, 'data_id')
        assert_given(num_levels, 'num_levels')
        super().__init__(data_id=data_id, type_specifier=type_specifier, **kwargs)
        self.num_levels = num_levels

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_MULTILEVEL_DATASET

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'MultiLevelDatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        d['num_levels'] = self.num_levels
        return d


class GeoDataFrameDescriptor(DataDescriptor):
    """
    A descriptor for a geo-vector dataset represented by a geopandas.GeoDataFrame instance.
    """

    def __init__(self,
                 data_id: str,
                 feature_schema: Any = None,
                 open_params_schema: JsonObjectSchema = None,
                 **kwargs):
        super().__init__(data_id=data_id,
                         type_specifier=TYPE_SPECIFIER_GEODATAFRAME,
                         open_params_schema=open_params_schema,
                         **kwargs)
        self.feature_schema = feature_schema

    def _assert_type_specifier(self, type_specifier: str):
        if not TYPE_SPECIFIER_GEODATAFRAME.is_satisfied_by(type_specifier):
            raise ValueError(f'type_specifier must be compatible with "{TYPE_SPECIFIER_GEODATAFRAME}" type specifier, '
                             f'was "{type_specifier}"')

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'MultiLevelDatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        _copy_none_null_props(self, d, ['feature_schema'])
        return d


def _convert_nans_to_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def _copy_none_null_props(obj: Any, d: Dict[str, Any], names: Sequence[str]):
    for name in names:
        value = getattr(obj, name)
        if value is not None:
            d[name] = value


register_json_formatter(DataDescriptor)
register_json_formatter(DatasetDescriptor)
register_json_formatter(VariableDescriptor)
register_json_formatter(MultiLevelDatasetDescriptor)
register_json_formatter(GeoDataFrameDescriptor)
