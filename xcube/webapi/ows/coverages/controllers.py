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
from xcube.util.timeindex import ensure_time_index_compatible
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

    ds = get_dataset(ctx, collection_id)
    native_crs = get_crs_from_dataset(ds)

    # See https://docs.ogc.org/DRAFTS/19-087.html#_parameter_crs
    final_crs = pyproj.CRS(query['crs'][0]) if 'crs' in query else native_crs

    # TODO: apply a size limit to avoid OOMs from attempting to
    #  produce arbitrarily large coverages

    if 'properties' in query:
        requested_vars = set(query['properties'][0].split(','))
        data_vars = set(map(str, ds.data_vars))
        unrecognized_vars = requested_vars - data_vars
        if unrecognized_vars == set():
            ds = ds.drop_vars(
                list(data_vars - requested_vars - {'crs', 'spatial_ref'})
            )
        else:
            raise ApiError.BadRequest(
                f'The following properties are not present in the coverage '
                f'{collection_id}: {", ".join(unrecognized_vars)}'
            )

    if 'datetime' in query:
        # TODO double-check that we don't need to support quotation marks
        if 'time' not in ds.variables:
            raise ApiError.BadRequest(
                f'"datetime" parameter invalid for coverage "{collection_id}",'
                'which has no "time" dimension.'
            )
        timespec = query['datetime'][0]
        if '/' in timespec:
            # Format: "<start>/<end>". ".." indicates an open range.
            time_limits = list(
                map(
                    lambda limit: None if limit == '..' else limit,
                    timespec.split('/'),
                )
            )
            time_slice = slice(*time_limits[:2])
            time_slice = ensure_time_index_compatible(ds, time_slice)
            ds = ds.sel(time=time_slice)
        else:
            timespec = ensure_time_index_compatible(ds, timespec)
            ds = ds.sel(time=timespec, method='nearest').squeeze()

    if 'subset' in query:
        ds = _apply_subsetting(
            ds, query['subset'][0], query.get('subset-crs', ['OGC:CRS84'])[0]
        )

    if 'bbox' in query:
        ds = _apply_bbox_2(
            ds, query['bbox'][0], query.get('bbox-crs', ['OGC:CRS84'])[0]
        )

    # ds.rio.write_crs(get_crs_from_dataset(ds), inplace=True)
    source_gm = GridMapping.from_dataset(ds, crs=get_crs_from_dataset(ds))
    target_gm = None

    if get_crs_from_dataset(ds) != final_crs:
        target_gm = source_gm.transform(final_crs).to_regular()
    if 'scale-factor' in query:
        scale_factor_spec = query['scale-factor'][0]
        try:
            scale_factor = float(scale_factor_spec)
        except ValueError:
            raise ApiError.BadRequest(
                f'Invalid scale-factor "{scale_factor_spec}"'
            )
        if target_gm is None:
            target_gm = source_gm
        target_gm = target_gm.scale(scale_factor)
    if 'scale-axes' in query:
        # TODO implement scale-axes
        raise ApiError.NotImplemented(
            'The scale-axes parameter is not yet supported.'
        )
    if 'scale-size' in query:
        # TODO implement scale-size
        raise ApiError.NotImplemented(
            'The scale-size parameter is not yet supported.'
        )

    if target_gm is not None:
        ds = resample_in_space(ds, source_gm=source_gm, target_gm=target_gm)

    # TODO: if final_crs == bbox-crs != native_crs, reapply the bbox here
    # since the original one (pre-transform) may have been too large.

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


def _apply_subsetting(
    ds: xr.Dataset, subset_spec: str, subset_crs: str
) -> xr.Dataset:
    indexers = _subset_to_indexers(subset_spec, ds)
    # TODO only reproject if there are spatial indexers
    # ds = _reproject_if_needed(ds, subset_crs)
    if indexers.indices:
        ds = ds.sel(indexers=indexers.indices, method='nearest')
    if indexers.slices:
        ds = ds.sel(indexers=indexers.slices)
    return ds


def _apply_bbox(ds: xr.Dataset, bbox_spec: str, bbox_crs: str) -> xr.Dataset:
    try:
        bbox = list(map(float, bbox_spec.split(',')))
    except ValueError:
        raise ApiError.BadRequest(f'Invalid bbox "{bbox_spec}"')
    if len(bbox) not in {4, 6}:
        raise ApiError.BadRequest(
            f'Invalid bbox "{bbox_spec}": must have 4 or 6 elements'
        )
    ds = _reproject_if_needed(ds, bbox_crs)
    dims = [
        d
        for d in map(str, ds.dims)
        if d.lower() in {'lat', 'lon', 'latitude', 'longitude', 'x', 'y'}
    ]
    bbox_ndims = len(bbox) // 2
    for i in range(min(len(dims), bbox_ndims)):
        dim = dims[i]
        min_, max_ = _correct_inverted_y_range(
            ds, dim, (bbox[i], bbox[i + bbox_ndims])
        )
        ds = ds.sel({dim: slice(min_, max_)})
    return ds


def _apply_bbox_2(ds: xr.Dataset, bbox_spec: str, bbox_crs: str) -> xr.Dataset:
    try:
        bbox = list(map(float, bbox_spec.split(',')))
    except ValueError:
        raise ApiError.BadRequest(f'Invalid bbox "{bbox_spec}"')
    if len(bbox) != 4:
        # TODO Handle 3D bounding boxes
        raise ApiError.BadRequest(
            f'Invalid bbox "{bbox_spec}": must have 4 elements'
        )
    crs_ds = get_crs_from_dataset(ds)
    crs_bbox = pyproj.CRS.from_string(bbox_crs)
    if crs_ds != crs_bbox:
        transformer = pyproj.Transformer.from_crs(
            crs_bbox, crs_ds  # always_xy=True
        )
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        bbox = transformer.transform_bounds(*bbox)
    h_dim = _get_h_dim(ds)
    v_dim = _get_v_dim(ds)
    v_slice = _correct_inverted_y_range(ds, v_dim, (bbox[1], bbox[3]))
    ds = ds.sel({h_dim: slice(bbox[0], bbox[2]), v_dim: slice(*v_slice)})
    return ds


def _get_h_dim(ds: xr.Dataset):
    return [
        d for d in list(map(str, ds.dims)) if d[:2].lower() in {'x', 'lon'}
    ][0]


def _get_v_dim(ds: xr.Dataset):
    return [
        d for d in list(map(str, ds.dims)) if d[:2].lower() in {'y', 'lat'}
    ][0]


def _reproject_if_needed(ds: xr.Dataset, target_crs: str):
    source_crs = get_crs_from_dataset(ds)
    if source_crs == pyproj.CRS(target_crs):
        return ds
    else:
        source_gm = GridMapping.from_dataset(ds).to_regular()
        target_gm_irregular = source_gm.transform(target_crs)
        target_gm = target_gm_irregular.to_regular()
        ds = resample_in_space(ds, source_gm=source_gm, target_gm=target_gm)
        if 'crs' not in ds.variables:
            ds['crs'] = 0
        ds.crs.attrs['spatial_ref'] = target_crs
        return ds


def _subset_to_indexers(subset_spec: str, ds: xr.Dataset) -> _IndexerTuple:
    indices, slices = {}, {}
    for part in subset_spec.split(','):
        # First try matching with quotation marks
        m = re.match(
            '^(.*)[(]"([^")]*)"(?::"(.*)")?[)]$', part
        )
        if m is None:
            # If that fails, try without quotation marks
            m = re.match(
                '^(.*)[(]([^:)]*)(?::(.*))?[)]$', part
            )
        if m is None:
            raise ApiError.BadRequest(
                f'Unrecognized subset specifier "{part}"'
            )
        else:
            axis, low, high = m.groups()
        if axis not in ds.dims:
            raise ApiError.BadRequest(f'Axis "{axis}" does not exist.')
        if high is None:
            if axis == 'time':
                indices[axis] = \
                    ensure_time_index_compatible(ds, low)
            else:
                try:
                    # Parse to float if possible
                    indices[axis] = float(low)
                except ValueError:
                    indices[axis] = low
        else:
            low = None if low == '*' else low
            high = None if high == '*' else high
            if axis == 'time':
                slices[axis] = \
                    ensure_time_index_compatible(ds, slice(low, high))
            else:
                # TODO Handle non-float arguments
                low = float(low)
                high = float(high)
                low, high = _correct_inverted_y_range(ds, axis, (low, high))
                slices[axis] = slice(low, high)

    return _IndexerTuple(indices, slices)


def _correct_inverted_y_range(
    ds: xr.Dataset, axis: str, range_: tuple[float, float]
) -> tuple[float, float]:
    x0, x1 = range_
    # Make sure latitude slice direction matches axis direction.
    # (For longitude, a descending-order slice is valid.)
    if (
        None not in range_
        and axis in {'lat', 'latitude', 'y'}
        and (x0 < x1) != (ds[axis][0] < ds[axis][-1])
    ):
        x0, x1 = x1, x0
    return x0, x1


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

    ds = ds.squeeze()

    with (tempfile.TemporaryDirectory() as tempdir):
        path = os.path.join(tempdir, 'out.' + image_format)
        ds = ds.drop_vars(names=['crs', 'spatial_ref'], errors='ignore').squeeze()
        if len(ds.data_vars) == 1:
            ds[list(ds.data_vars)[0]].rio.to_raster(path)
        else:
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
        srsName=get_crs_from_dataset(ds).to_string(),
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
        lower_bound, upper_bound = axis[0].item(), axis[-1].item()
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


def get_crs_from_dataset(ds: xr.Dataset) -> pyproj.crs.CRS:
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
            for attr_name in 'spatial_ref', 'crs_wkt':
                if attr_name in var.attrs:
                    crs_string = ds[var_name].attrs[attr_name]
                    return pyproj.CRS(crs_string)
    return pyproj.CRS('EPSG:4326')


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
        'srsName': get_crs_from_dataset(ds).to_string(),
        'axisLabels': list(ds.dims.keys()),
        'axis': _get_axes_properties(ds),
    }
