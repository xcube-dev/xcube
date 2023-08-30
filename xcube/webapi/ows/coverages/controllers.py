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
from xcube.server.api import ApiRequest
from xcube.webapi.datasets.context import DatasetsContext
import os
import tempfile
import xarray as xr
import pyproj
import numpy as np
import rasterio


def get_coverage_as_json(ctx: DatasetsContext, collection_id: str):
    return {
        'id': collection_id,
        'type': 'CoverageByDomainAndRange',
        'envelope': get_collection_envelope(ctx, collection_id),
        'domainSet': get_coverage_domainset(ctx, collection_id),
        'rangeSet': {
            'type': 'RangeSet',
            'dataBlock': {'type': 'VDataBlock', 'values': ['string']},  # FIXME
        },
        'rangeType': get_coverage_rangetype(ctx, collection_id),
        'metadata': get_collection_metadata(ctx, collection_id),
    }


def get_coverage_data(
    ctx: DatasetsContext,
    collection_id: str,
    request: ApiRequest,
    content_type: str,
):
    query = request.query
    ds = get_dataset(ctx, collection_id)
    if 'bbox' in query:
        bbox = list(map(float, query['bbox'][0].split(',')))
        ds = ds.sel(lat=slice(bbox[0], bbox[2]), lon=slice(bbox[1], bbox[3]))
    if 'datetime' in query:
        timespec = query['datetime']
        if '/' in timespec:
            timefrom, timeto = timespec[0].split('/')
            ds = ds.sel(time=slice(timefrom, timeto))
        else:
            ds = ds.sel(time=timespec, method='nearest').squeeze()
    if 'properties' in query:
        vars_to_keep = set(query['properties'][0].split(','))
        data_vars = set(ds.data_vars)
        vars_to_drop = list(data_vars - vars_to_keep)
        ds = ds.drop_vars(vars_to_drop)
    if content_type in ['image/tiff', 'application/x-geotiff']:
        return dataset_to_tiff(ds)
    if content_type in ['application/netcdf', 'application/x-netcdf']:
        return dataset_to_netcdf(ds)
    return None


def dataset_to_tiff(ds: xr.Dataset):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'out.tiff')
        ds.rio.to_raster(path)
        with open(path, 'rb') as fh:
            data = fh.read()
    return data


def dataset_to_netcdf(ds: xr.Dataset):
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'out.nc')
        ds.to_netcdf(path)
        with open(path, 'rb') as fh:
            data = fh.read()
    return data


def get_coverage_domainset(ctx: DatasetsContext, collection_id: str):
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
        axis=get_axes_properties(ds),
        gridLimits=grid_limits,
    )
    domain_set = dict(type='DomainSet', generalGrid=grid)
    return domain_set


def get_collection_metadata(ctx: DatasetsContext, collection_id: str):
    ds = get_dataset(ctx, collection_id)
    return ds.attrs


def get_dataset(ctx, collection_id):
    ml_dataset = ctx.get_ml_dataset(collection_id)
    ds = ml_dataset.get_dataset(0)
    assert isinstance(ds, xr.Dataset)
    return ds


def get_axes_properties(ds: xr.Dataset) -> list[dict]:
    return [_get_axis_properties(ds, dim) for dim in ds.dims]


def _get_axis_properties(ds: xr.Dataset, dim: str) -> dict:
    axis = ds.coords[dim]
    return dict(
        type='RegularAxis',
        axisLabel=dim,
        lowerBound=axis[0].item(),
        upperBound=axis[-1].item(),
        resolution=abs((axis[-1] - axis[0]).item() / len(axis)),
        uomLabel=_get_units(ds, dim),
    )


def _get_grid_limits_axis(ds: xr.Dataset, dim: str):
    return dict(
        type='IndexAxis', axisLabel=dim, lowerBound=0, upperBound=len(ds[dim])
    )


def _get_units(ds: xr.Dataset, dim: str):
    coord = ds.coords[dim]
    if hasattr(coord, 'attrs') and 'units' in coord.attrs:
        return coord.attrs['units']
    else:
        return 'degrees'


def get_crs_from_dataset(ds: xr.Dataset):
    for var_name in 'crs', 'spatial_ref':
        if var_name in ds.variables:
            var = ds[var_name]
            if 'spatial_ref' in var.attrs:
                crs_string = ds[var_name].attrs['spatial_ref']
                return pyproj.crs.CRS(crs_string).to_string()
    return 'EPSG:4326'


def get_coverage_rangetype(ctx: DatasetsContext, collection_id: str):
    ds = get_dataset(ctx, collection_id)
    result = dict(type='DataRecord', field=[])
    for var_name in ds.data_vars:
        result['field'].append(
            dict(
                type='Quantity',
                name=var_name,
                description=_get_var_description(ds[var_name]),
                encodingInfo=dict(
                    dataType=_dtype_to_opengis_datatype(ds[var_name].dtype)
                ),
            )
        )
    return result


def _dtype_to_opengis_datatype(dt: np.dtype):
    nbits = 4 * dt.itemsize
    int_size_map = {8: 'Byte', 16: 'Short', 32: 'Int', 64: 'Long'}
    if np.issubdtype(dt, np.floating):
        opengis_type = f'float{nbits}'
    elif np.issubdtype(dt, np.signedinteger):
        opengis_type = f'Signed{int_size_map[nbits]}'
    elif np.issubdtype(dt, np.unsignedinteger):
        opengis_type = f'Unsigned{int_size_map[nbits]}'
    elif 'datetime64' in str(dt):
        opengis_type = 'http://www.opengis.net/def/bipm/UTC'
    else:
        opengis_type = ''  # TODO decide what to do in this case
    return 'http://www.opengis.net/def/dataType/OGC/0/' + opengis_type


def _get_var_description(var):
    if hasattr(var, 'attrs'):
        for attr in ['description', 'long_name', 'standard_name', 'name']:
            if attr in var.attrs:
                return var.attrs[attr]
    return var.name


async def get_collection_envelope(ds_ctx, collection_id):
    ds = get_dataset(ds_ctx, collection_id)
    return {
        'type': 'EnvelopeByAxis',
        'srsName': get_crs_from_dataset(ds),
        'axisLabels': list(ds.dims.keys()),
        'axis': get_axes_properties(ds),
    }
