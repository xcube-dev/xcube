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

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.core.geom import get_dataset_bounds
from xcube.core.timecoord import get_end_time_from_attrs
from xcube.core.timecoord import get_start_time_from_attrs
from xcube.core.timecoord import get_time_range_from_data
from xcube.core.store.typespecifier import TYPE_SPECIFIER_ANY
from xcube.core.store.typespecifier import TYPE_SPECIFIER_DATASET
from xcube.core.store.typespecifier import TYPE_SPECIFIER_GEODATAFRAME
from xcube.core.store.typespecifier import TYPE_SPECIFIER_MULTILEVEL_DATASET
from xcube.core.store.typespecifier import TypeSpecifier
from xcube.core.store.typespecifier import get_type_specifier
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.ipython import register_json_formatter
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonDateSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema


# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.schema.CubeSchema class
#   xcube.webapi.context.DatasetDescriptor type
#   responses of xcube.webapi.controllers.catalogue
# TODO: write tests
# TODO: validate params


def new_data_descriptor(data_id: str, data: Any, require: bool = False) -> 'DataDescriptor':
    if isinstance(data, xr.Dataset):
        coords = _build_variable_descriptor_dict(data.coords)
        data_vars = _build_variable_descriptor_dict(data.data_vars)
        spatial_res = _determine_spatial_res(data)
        bbox = _determine_bbox(data)
        time_coverage_start, time_coverage_end = _determine_time_coverage(data)
        time_period = _determine_time_period(data)
        return DatasetDescriptor(data_id=data_id,
                                 type_specifier=get_type_specifier(data),
                                 bbox=bbox,
                                 time_range=(time_coverage_start, time_coverage_end),
                                 time_period=time_period,
                                 spatial_res=spatial_res,
                                 coords=coords,
                                 dims=dict(data.dims),
                                 data_vars=data_vars,
                                 attrs=data.attrs)
    elif isinstance(data, MultiLevelDataset):
        # TODO: implement me: data -> MultiLevelDatasetDescriptor
        return MultiLevelDatasetDescriptor(data_id=data_id, num_levels=5)
    elif isinstance(data, gpd.GeoDataFrame):
        # TODO: implement me: data -> GeoDataFrameDescriptor
        return GeoDataFrameDescriptor(data_id=data_id)
    elif not require:
        return DataDescriptor(data_id=data_id, type_specifier=TYPE_SPECIFIER_ANY)
    raise NotImplementedError()


def _build_variable_descriptor_dict(variables) -> Mapping[str, 'VariableDescriptor']:
    return {str(var_name): VariableDescriptor(
        name=str(var_name),
        dtype=str(var.dtype),
        dims=var.dims,
        chunks=tuple([max(chunk) for chunk in tuple(var.chunks)]) if var.chunks else None,
        attrs=var.attrs if var.attrs else None)
        for var_name, var in variables.items()}


def _determine_bbox(data: xr.Dataset) -> Optional[Tuple[float, float, float, float]]:
    try:
        return get_dataset_bounds(data)
    except ValueError:
        if 'geospatial_lat_min' in data.attrs and \
                'geospatial_lon_min' in data.attrs and \
                'geospatial_lat_max' in data.attrs and \
                'geospatial_lon_max' in data.attrs:
            return (data.geospatial_lat_min, data.geospatial_lon_min,
                    data.geospatial_lat_max, data.geospatial_lon_max)


def _determine_spatial_res(data: xr.Dataset):
    # TODO get rid of these hard-coded coord names as soon as new resampling is available
    lat_dimensions = ['lat', 'latitude', 'y']
    for lat_dimension in lat_dimensions:
        if lat_dimension in data:
            lat_diff = data[lat_dimension].diff(dim=data[lat_dimension].dims[0]).values
            lat_res = lat_diff[0]
            lat_regular = np.allclose(lat_res, lat_diff, 1e-8)
            if lat_regular:
                return abs(lat_res)


def _determine_time_coverage(data: xr.Dataset):
    start_time, end_time = get_time_range_from_data(data)
    if start_time is None:
        start_time = get_start_time_from_attrs(data)
    else:
        start_time = pd.to_datetime(start_time).isoformat()
    if end_time is None:
        end_time = get_end_time_from_attrs(data)
    else:
        end_time = pd.to_datetime(end_time).isoformat()
    return start_time, end_time


def _determine_time_period(data: xr.Dataset):
    if 'time' in data and len(data['time'].values) > 0:
        time_diff = data['time'].diff(dim=data['time'].dims[0]).values.astype(np.float64)
        time_res = time_diff[0]
        time_regular = np.allclose(time_res, time_diff, 1e-8)
        if time_regular:
            time_period = pd.to_timedelta(time_res).isoformat()
            # remove leading P
            time_period = time_period[1:]
            # removing sub-day precision
            return time_period.split('T')[0]


class DataDescriptor:
    """
    A generic descriptor for any data.
    Also serves as a base class for more specific data descriptors.

    :param data_id: An identifier for the data
    :param type_specifier: A type specifier for the data
    :param crs: A coordinate reference system identifier, as an EPSG, PROJ or WKT string
    :param bbox: A bounding box of the data
    :param time_range: Start and end time delimiting this data's temporal extent
    :param time_period: The data's periodicity if it is evenly temporally resolved.
    :param open_params_schema: A JSON schema describing the parameters that may be used to open
    this data.
    """

    def __init__(self,
                 data_id: str,
                 type_specifier: Union[str, TypeSpecifier],
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 open_params_schema: JsonObjectSchema = None):
        assert_given(data_id, 'data_id')
        self._assert_valid_type_specifier(type_specifier)
        self.data_id = data_id
        self.type_specifier = TypeSpecifier.normalize(type_specifier)
        self.crs = crs
        self.bbox = tuple(bbox) if bbox else None
        self.time_range = tuple(time_range) if time_range else None
        self.time_period = time_period
        self.open_params_schema = open_params_schema

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DataDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('data_id', d)
        assert_in('type_specifier', d)
        if TYPE_SPECIFIER_DATASET.is_satisfied_by(d['type_specifier']):
            return DatasetDescriptor.from_dict(d)
        elif TYPE_SPECIFIER_GEODATAFRAME.is_satisfied_by(d['type_specifier']):
            return GeoDataFrameDescriptor.from_dict(d)
        return DataDescriptor(data_id=d['data_id'],
                              type_specifier=d['type_specifier'],
                              crs=d.get('crs', None),
                              bbox=d.get('bbox', None),
                              time_range=d.get('time_range', None),
                              time_period=d.get('time_period', None),
                              open_params_schema=d.get('open_params_schema', None))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict(type_specifier=str(self.type_specifier))
        _copy_none_null_props(self, d, ['data_id', 'crs', 'bbox', 'time_range', 'time_period'])

        if self.open_params_schema is not None:
            d['open_params_schema'] = self.open_params_schema.to_dict()
        return d

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_ANY

    @classmethod
    def _assert_valid_type_specifier(cls, type_specifier: Optional[str]):
        assert_given(type_specifier, 'type_specifier')
        base_type_specifier = cls._get_base_type_specifier()
        if not base_type_specifier.is_satisfied_by(type_specifier):
            raise ValueError('type_specifier must satisfy'
                             f' type specifier "{base_type_specifier}", but was "{type_specifier}"')


class DatasetDescriptor(DataDescriptor):
    """
    A descriptor for a gridded, N-dimensional dataset represented by xarray.Dataset.
    Comprises a description of the data variables contained in the dataset.

    :param data_id: An identifier for the data
    :param type_specifier: A type specifier for the data
    :param crs: A coordinate reference system identifier, as an EPSG, PROJ or WKT string
    :param bbox: A bounding box of the data
    :param time_range: Start and end time delimiting this data's temporal extent (see
    https://github.com/dcs4cop/xcube/blob/master/docs/source/storeconv.md#date-time-and-duration-specifications )
    :param time_period: The data's periodicity if it is evenly temporally resolved (see
    https://github.com/dcs4cop/xcube/blob/master/docs/source/storeconv.md#date-time-and-duration-specifications )
    :param spatial_res: The spatial extent of a pixel in crs units
    :param dims: A mapping of the dataset's dimensions to their sizes
    :param coords: mapping of the dataset's data coordinate names to VariableDescriptors
    (``xcube.core.store.VariableDescriptor``).
    :param data_vars: A mapping of the dataset's variable names to VariableDescriptors
    (``xcube.core.store.VariableDescriptor``).
    :param attrs: A mapping containing arbitrary attributes of the dataset
    :param open_params_schema: A JSON schema describing the parameters that may be used to open
    this data
    """

    def __init__(self,
                 data_id: str,
                 type_specifier: Union[str, TypeSpecifier] = TYPE_SPECIFIER_DATASET,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None,
                 spatial_res: float = None,
                 dims: Mapping[str, int] = None,
                 coords: Mapping[str, 'VariableDescriptor'] = None,
                 data_vars: Mapping[str, 'VariableDescriptor'] = None,
                 attrs: Mapping[str, any] = None,
                 open_params_schema: JsonObjectSchema = None):
        super().__init__(data_id=data_id,
                         type_specifier=type_specifier,
                         crs=crs,
                         bbox=bbox,
                         time_range=time_range,
                         time_period=time_period,
                         open_params_schema=open_params_schema)
        self.dims = dict(dims) if dims else None
        self.coords = coords if coords else None
        self.data_vars = data_vars if data_vars else None
        self.spatial_res = spatial_res
        self.attrs = _convert_nans_to_none(dict(attrs)) if attrs else None

    @classmethod
    def get_schema(cls) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                data_id=JsonStringSchema(min_length=1),
                type_specifier=JsonStringSchema(default=TYPE_SPECIFIER_DATASET, min_length=1),
                crs=JsonStringSchema(min_length=1),
                bbox=JsonArraySchema(items=[JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema(),
                                            JsonNumberSchema()]),
                time_range=JsonDateSchema.new_range(nullable=True),
                time_period=JsonStringSchema(min_length=1),
                spatial_res=JsonNumberSchema(exclusive_minimum=0.0),
                dims=JsonObjectSchema(additional_properties=True),
                coords=JsonObjectSchema(additional_properties=True),
                data_vars=JsonObjectSchema(additional_properties=True),
                attrs=JsonObjectSchema(additional_properties=True),
                open_params_schema=JsonObjectSchema(additional_properties=True),
            ),
            required=['data_id'],
            additional_properties=False,
            factory=cls)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('data_id', d)
        coords = d.get('coords')
        if coords:
            coords = {k: VariableDescriptor.from_dict(v) for k, v in coords.items()}
        data_vars = d.get('data_vars')
        if data_vars:
            data_vars = {k: VariableDescriptor.from_dict(v) for k, v in data_vars.items()}
        return DatasetDescriptor(data_id=d['data_id'],
                                 type_specifier=d.get('type_specifier', TYPE_SPECIFIER_DATASET),
                                 crs=d.get('crs'),
                                 bbox=d.get('bbox'),
                                 time_range=d.get('time_range'),
                                 time_period=d.get('time_period'),
                                 spatial_res=d.get('spatial_res'),
                                 coords=coords,
                                 dims=d.get('dims'),
                                 data_vars=data_vars,
                                 attrs=d.get('attrs'),
                                 open_params_schema=d.get('open_params_schema'))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        if self.coords is not None:
            coords = {k: v.to_dict() for k, v in self.coords.items()}
            d['coords'] = coords
        if self.data_vars is not None:
            data_vars = {k: v.to_dict() for k, v in self.data_vars.items()}
            d['data_vars'] = data_vars
        _copy_none_null_props(self, d, ['spatial_res', 'dims', 'attrs'])
        return d

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_DATASET


class VariableDescriptor:
    """
    A descriptor for dataset variable represented by xarray.DataArray instances.
    They are part of dataset descriptor for an gridded, N-dimensional dataset represented by
    xarray.Dataset.

    :param name: The variable name
    :param dtype: The data type of the variable.
    :param dims: A list of the names of the variable's dimensions.
    :param chunks: A list of the chunk sizes of the variable's dimensions
    :param attrs: A mapping containing arbitrary attributes of the variable
    """

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 chunks: Sequence[int] = None,
                 attrs: Mapping[str, any] = None):
        assert_given(name, 'name')
        assert_given(dtype, 'dtype')
        assert_given(dims, 'dims')
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.chunks = tuple(chunks) if chunks else None
        self.ndim = len(self.dims)
        self.attrs = _convert_nans_to_none(dict(attrs)) if attrs is not None else None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('name', d)
        assert_in('dtype', d)
        assert_in('dims', d)
        return VariableDescriptor(d['name'],
                                  d['dtype'],
                                  d['dims'],
                                  d.get('chunks'),
                                  d.get('attrs', None))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict(name=self.name, dtype=self.dtype)
        _copy_none_null_props(self, d, ['dims', 'ndim', 'chunks', 'attrs'])
        return d


class MultiLevelDatasetDescriptor(DatasetDescriptor):
    """
    A descriptor for a gridded, N-dimensional, multi-level, multi-resolution dataset represented by
    xcube.core.mldataset.MultiLevelDataset.

    :param data_id: An identifier of the multi-level dataset
    :param num_levels: The number of levels of this multi-level dataset
    :param type_specifier: A type specifier for the multi-level dataset
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
    def from_dict(cls, d: Mapping[str, Any]) -> 'MultiLevelDatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        d['num_levels'] = self.num_levels
        return d

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_MULTILEVEL_DATASET


class GeoDataFrameDescriptor(DataDescriptor):
    """
    A descriptor for a geo-vector dataset represented by a geopandas.GeoDataFrame instance.

    :param data_id: An identifier of the geopandas.GeoDataFrame
    :param feature_schema: A schema describing the properties of the vector data
    :param open_params_schema: A JSON schema describing the parameters that may be used to open
    this geopandas.GeoDataFrame
    """

    def __init__(self,
                 data_id: str,
                 type_specifier=TYPE_SPECIFIER_GEODATAFRAME,
                 feature_schema: Any = None,
                 open_params_schema: JsonObjectSchema = None,
                 **kwargs):
        super().__init__(data_id=data_id,
                         type_specifier=type_specifier,
                         open_params_schema=open_params_schema,
                         **kwargs)
        self.feature_schema = feature_schema

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'GeoDataFrameDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        assert_in('data_id', d)
        return GeoDataFrameDescriptor(data_id=d['data_id'],
                                      type_specifier=d.get('type_specifier',
                                                           TYPE_SPECIFIER_GEODATAFRAME),
                                      open_params_schema=d.get('open_params_schema', None))

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        _copy_none_null_props(self, d, ['feature_schema'])
        return d

    @classmethod
    def _get_base_type_specifier(cls) -> TypeSpecifier:
        return TYPE_SPECIFIER_GEODATAFRAME


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
