# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import re
import tempfile
from typing import Mapping, Sequence, Optional, Any, Literal, NamedTuple

import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space
from xcube.server.api import ApiError
from xcube.webapi.datasets.context import DatasetsContext


def get_coverage_as_json(ctx: DatasetsContext, collection_id: str):
    """
    Return a JSON representation of the specified coverage

    Currently, the range set component is omitted.

    :param ctx: a dataset context
    :param collection_id: the ID of the collection providing the coverage
    :return: a JSON representation of the coverage
    """
    return {
        'id': collection_id,
        'type': 'CoverageByDomainAndRange',
        'envelope': get_collection_envelope(ctx, collection_id),
        'domainSet': get_coverage_domainset(ctx, collection_id),
        'rangeSet': {
            'type': 'RangeSet',
            # TODO: Wait for next update to API specification before
            #  implementing the data block -- not clear yet whether this
            #  is being deprecated along with the rangeSet endpoint.
            'dataBlock': {'type': 'VDataBlock', 'values': ['TODO']},
        },
        'rangeType': get_coverage_rangetype(ctx, collection_id),
        'metadata': get_collection_metadata(ctx, collection_id),
    }


def get_coverage_data(
    ctx: DatasetsContext,
    collection_id: str,
    query: Mapping[str, Sequence[str]],
    content_type: str,
) -> Optional[bytes]:
    """
    Return coverage data from a dataset

    This method currently returns coverage data from a dataset as either
    TIFF or NetCDF. The bbox, datetime, and properties parameters are
    currently handled.

    :param ctx: a datasets context
    :param collection_id: the dataset from which to return the coverage
    :param query: the HTTP query parameters
    :param content_type: the MIME type of the desired output format
    :return: the coverage as bytes in the requested output format, or None
             if the requested output format is not supported
    """

    # TODO: support scale-factor, scale-axes, scale-size, subset-crs,
    #  and bbox-crs

    ds = get_dataset(ctx, collection_id)
    if 'subset' in query:
        indexers = _subset_to_indexers(query['subset'][0], ds)
        if indexers.indices:
            ds = ds.sel(indexers=indexers.indices, method='nearest')
        if indexers.slices:
            ds = ds.sel(indexers=indexers.slices)
    if 'bbox' in query:
        # TODO: flip latitude if required
        # TODO: slice according to axis order in dataset
        bbox = list(map(float, query['bbox'][0].split(',')))
        if {'lat', 'lon'}.issubset(ds.variables):
            ds = ds.sel(
                lat=slice(bbox[0], bbox[2]), lon=slice(bbox[1], bbox[3])
            )
        else:
            ds = ds.sel(y=slice(bbox[0], bbox[2]), x=slice(bbox[1], bbox[3]))
    if 'datetime' in query:
        # TODO: Raise an exception for non-existent time axis
        # TODO: Support open ranges
        timespec = query['datetime'][0]
        if '/' in timespec:
            timefrom, timeto = timespec.split('/')
            ds = ds.sel(time=slice(timefrom, timeto))
        else:
            ds = ds.sel(time=timespec, method='nearest').squeeze()
    if 'properties' in query:
        # TODO: Raise an exception for non-existent properties
        vars_to_keep = set(query['properties'][0].split(','))
        data_vars = set(ds.data_vars)
        vars_to_drop = list(data_vars - vars_to_keep)
        ds = ds.drop_vars(vars_to_drop)
    if 'crs' in query:
        target_crs = query['crs'][0]
        source_gm = GridMapping.from_dataset(ds)
        target_gm = source_gm.transform(target_crs).to_regular()
        ds = resample_in_space(ds, source_gm=source_gm, target_gm=target_gm)

    media_types = dict(
        tiff={'geotiff', 'image/tiff', 'application/x-geotiff'},
        png={'png', 'image/png'},
        netcdf={'netcdf', 'application/netcdf', 'application/x-netcdf'},
    )
    if content_type in media_types['tiff']:
        return dataset_to_image(ds, 'tiff')
    elif content_type in media_types['png']:
        return dataset_to_image(ds, 'png')
    elif content_type in media_types['netcdf']:
        return dataset_to_netcdf(ds)
    else:
        # It's expected that the caller (server API handler) will catch
        # unhandled types, but we may as well do the right thing if any
        # do slip through.
        raise ApiError.UnsupportedMediaType(
            f'Unsupported media type {content_type}. '
            + 'Available media types: '
            + ', '.join(
                [type_ for value in media_types.values() for type_ in value]
            )
        )


_IndexerTuple = NamedTuple(
    'Indexers', [('indices', dict[str, Any]), ('slices', dict[str, slice])]
)


def _subset_to_indexers(subset_spec: str, ds: xr.Dataset) -> _IndexerTuple:
    indices, slices = {}, {}
    for part in subset_spec.split(','):
        axis, low, high = re.match(
            '^(.*)[(]([^:)]*)(?::(.*))?[)]$', part
        ).groups()
        if high is None:
            # TODO: parse to float if possible
            indices[axis] = low
        else:
            # TODO: don't flip longitude
            if (low < high) != (ds[axis][0] < ds[axis][-1]):
                low, high = high, low
            if axis != 'time':
                # TODO: catch parse exception and pass through as string
                low = float(low)
                high = float(high)
            slices[axis] = slice(
                None if low == '*' else low, None if high == '*' else high
            )
    return _IndexerTuple(indices, slices)


def dataset_to_image(
    ds: xr.Dataset, image_format: Literal['png', 'tiff'] = 'png'
) -> bytes:
    """
    Return an in-memory bitmap (TIFF or PNG) representing a dataset

    :param ds: a dataset
    :param image_format: image format to generate ("png" or "tiff")
    :return: TIFF-formatted bytes representing the dataset
    """

    if image_format == 'png':
        for var in ds.data_vars:
            # rasterio's PNG driver only supports these data types.
            if ds[var].dtype not in {np.uint8, np.uint16}:
                ds[var] = ds[var].astype(np.uint16, casting='unsafe')

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'out.' + image_format)
        ds.rio.to_raster(path)
        with open(path, 'rb') as fh:
            data = fh.read()
    return data


def dataset_to_netcdf(ds: xr.Dataset) -> bytes:
    """
    Return an in-memory NetCDF representing a dataset

    :param ds: a dataset
    :return: NetCDF-formatted bytes representing the dataset
    """
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'out.nc')
        ds.to_netcdf(path)
        with open(path, 'rb') as fh:
            data = fh.read()
    return data


def get_coverage_domainset(ctx: DatasetsContext, collection_id: str):
    """
    Return the domain set of a dataset-backed coverage

    The domain set is the set of input parameters (e.g. geographical extent,
    time span) for which a coverage is defined.

    :param ctx: a datasets context
    :param collection_id: the dataset for which to return the domain set
    :return: a dictionary representing an OGC API - Coverages domain set
    """
    ds = get_dataset(ctx, collection_id)
    grid_limits = dict(
        type='GridLimits',
        srsName=f'http://www.opengis.net/def/crs/OGC/0/Index{len(ds.dims)}D',
        axisLabels=list(ds.dims),
        axis=[_get_grid_limits_axis(ds, dim) for dim in ds.dims],
    )
    grid = dict(
        type='GeneralGridCoverage',
        srsName=get_crs_from_dataset(ds),
        axisLabels=list(ds.dims.keys()),
        axis=_get_axes_properties(ds),
        gridLimits=grid_limits,
    )
    return dict(type='DomainSet', generalGrid=grid)


def get_collection_metadata(ctx: DatasetsContext, collection_id: str):
    """
    Return a metadata dictionary for a dataset

    The metadata is taken directly from the dataset attributes.

    :param ctx: a datasets context
    :param collection_id: the dataset for which to return the metadata
    :return: a dictionary of metadata keys and values
    """
    ds = get_dataset(ctx, collection_id)
    return ds.attrs


def get_dataset(ctx: DatasetsContext, collection_id: str):
    """
    Get a dataset from a datasets context

    :param ctx: a datasets context
    :param collection_id: the ID of a dataset in the context
    :return: the dataset
    """
    ml_dataset = ctx.get_ml_dataset(collection_id)
    ds = ml_dataset.get_dataset(0)
    assert isinstance(ds, xr.Dataset)
    return ds


def _get_axes_properties(ds: xr.Dataset) -> list[dict]:
    return [_get_axis_properties(ds, dim) for dim in ds.dims]


def _get_axis_properties(ds: xr.Dataset, dim: str) -> dict[str, Any]:
    axis = ds.coords[dim]
    if np.issubdtype(axis.dtype, np.datetime64):
        lower_bound = np.datetime_as_string(axis[0])
        upper_bound = np.datetime_as_string(axis[-1])
    else:
        lower_bound, upper_bound = axis[0].item(), axis[-1].item(),
    return dict(
        type='RegularAxis',
        axisLabel=dim,
        lowerBound=lower_bound,
        upperBound=upper_bound,
        resolution=abs((axis[-1] - axis[0]).item() / len(axis)),
        uomLabel=get_units(ds, dim),
    )


def _get_grid_limits_axis(ds: xr.Dataset, dim: str) -> dict[str, Any]:
    return dict(
        type='IndexAxis', axisLabel=dim, lowerBound=0, upperBound=len(ds[dim])
    )


def get_units(ds: xr.Dataset, dim: str) -> str:
    coord = ds.coords[dim]
    if hasattr(coord, 'attrs') and 'units' in coord.attrs:
        return coord.attrs['units']
    if np.issubdtype(coord, np.datetime64):
        return np.datetime_data(coord)[0]
    # TODO: as a fallback for spatial axes, we could try matching dimensions
    #  to CRS axes and take the unit from the CRS definition.
    return 'unknown'


def get_crs_from_dataset(ds: xr.Dataset) -> str:
    """
    Return the CRS of a dataset as a string. The CRS is taken from the
    metadata of the crs or spatial_ref variables, if available.
    "EPSG:4326" is used as a fallback.

    :param ds: a dataset
    :return: a string representation of the dataset's CRS, or "EPSG:4326"
             if the CRS cannot be determined
    """
    for var_name in 'crs', 'spatial_ref':
        if var_name in ds.variables:
            var = ds[var_name]
            if 'spatial_ref' in var.attrs:
                crs_string = ds[var_name].attrs['spatial_ref']
                return pyproj.crs.CRS(crs_string).to_string()
    return 'EPSG:4326'


def get_coverage_rangetype(
    ctx: DatasetsContext, collection_id: str
) -> dict[str, list]:
    """
    Return the range type of a dataset

    The range type describes the data types of the dataset's variables
    """
    ds = get_dataset(ctx, collection_id)
    result = dict(type='DataRecord', field=[])
    for var_name in ds.data_vars:
        result['field'].append(
            dict(
                type='Quantity',
                name=var_name,
                description=get_dataarray_description(ds[var_name]),
                encodingInfo=dict(
                    dataType=dtype_to_opengis_datatype(ds[var_name].dtype)
                ),
            )
        )
    return result


def dtype_to_opengis_datatype(dt: np.dtype) -> str:
    """
    Convert a NumPy dtype to an equivalent OpenGIS type identifier string.

    :param dt: a NumPy dtype
    :return: an equivalent OpenGIS type identifier string, or an empty string
             if the dtype is not recognized
    """
    nbits = 8 * np.dtype(dt).itemsize
    int_size_map = {8: 'Byte', 16: 'Short', 32: 'Int', 64: 'Long'}
    prefix = 'http://www.opengis.net/def/dataType/OGC/0/'
    if np.issubdtype(dt, np.floating):
        opengis_type = f'{prefix}float{nbits}'
    elif np.issubdtype(dt, np.signedinteger):
        opengis_type = f'{prefix}signed{int_size_map[nbits]}'
    elif np.issubdtype(dt, np.unsignedinteger):
        opengis_type = f'{prefix}unsigned{int_size_map[nbits]}'
    elif 'datetime64' in str(dt):
        opengis_type = 'http://www.opengis.net/def/bipm/UTC'
    else:
        opengis_type = ''  # TODO decide what to do in this case
    return opengis_type


def get_dataarray_description(da: xr.DataArray) -> str:
    """
    Return a string describing a DataArray, either from an attribute or,
    as a fallback, from its name attribute.

    :param da: a DataArray
    :return: a string describing the DataArray
    """
    if hasattr(da, 'attrs'):
        for attr in ['description', 'long_name', 'standard_name', 'name']:
            if attr in da.attrs:
                return da.attrs[attr]
    return str(da.name)


def get_collection_envelope(ds_ctx, collection_id):
    """
    Return the OGC API - Coverages envelope of a dataset.

    The envelope comprises the extents of all the dataset's dimensions.

    :param ds_ctx: a datasets context
    :param collection_id: a dataset ID within the given context
    :return: the envelope of the specified dataset
    """
    ds = get_dataset(ds_ctx, collection_id)
    return {
        'type': 'EnvelopeByAxis',
        'srsName': get_crs_from_dataset(ds),
        'axisLabels': list(ds.dims.keys()),
        'axis': _get_axes_properties(ds),
    }
