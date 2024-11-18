# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
import os
import re
import tempfile
from typing import Optional, Any, Literal, NamedTuple, Union
from collections.abc import Mapping, Sequence

import numpy as np
import pyproj
import rasterio
import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.resampling import resample_in_space
from xcube.server.api import ApiError
from xcube.util.timeindex import ensure_time_index_compatible
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.ows.coverages.request import CoverageRequest
from xcube.webapi.ows.coverages.scaling import CoverageScaling
from xcube.webapi.ows.coverages.util import get_h_dim, get_v_dim


def get_coverage_as_json(ctx: DatasetsContext, collection_id: str):
    """Return a JSON representation of the specified coverage

    Currently, the range set component is omitted.

    Args:
        ctx: a dataset context
        collection_id: the ID of the collection providing the coverage

    Returns:
        a JSON representation of the coverage
    """
    return {
        "id": collection_id,
        "type": "CoverageByDomainAndRange",
        "envelope": get_collection_envelope(ctx, collection_id),
        "domainSet": get_coverage_domainset(ctx, collection_id),
        "rangeSet": {
            "type": "RangeSet",
            # TODO: Wait for next update to API specification before
            #  implementing the data block -- not clear yet whether this
            #  is being deprecated along with the rangeSet endpoint.
            "dataBlock": {"type": "VDataBlock", "values": ["TODO"]},
        },
        "rangeType": get_coverage_rangetype(ctx, collection_id),
        "metadata": get_collection_metadata(ctx, collection_id),
    }


def get_coverage_data(
    ctx: DatasetsContext,
    collection_id: str,
    query: Mapping[str, Sequence[str]],
    content_type: str,
) -> tuple[Optional[bytes], list[float], pyproj.CRS]:
    """Return coverage data from a dataset

    This method currently returns coverage data from a dataset as either
    TIFF or NetCDF. The bbox, datetime, and properties parameters are
    currently handled.

    Args:
        ctx: a datasets context
        collection_id: the dataset from which to return the coverage
        query: the HTTP query parameters
        content_type: the MIME type of the desired output format

    Returns:
        A tuple consisting of: (1) the coverage as bytes in the
        requested output format, or None if the requested output format
        is not supported; (2) the bounding box of the returned data,
        respecting the axis ordering of the CRS (e.g. latitude first for
        EPSG:4326); (3) the CRS of the returned dataset and bounding box
    """

    ds = get_dataset(ctx, collection_id)

    try:
        request = CoverageRequest(query)
    except ValueError as e:
        raise ApiError.BadRequest(str(e))

    # See https://docs.ogc.org/DRAFTS/19-087.html
    native_crs = get_crs_from_dataset(ds)
    final_crs = request.crs if request.crs is not None else native_crs
    bbox_crs = request.bbox_crs
    subset_crs = request.subset_crs

    if request.properties is not None:
        ds = _apply_properties(collection_id, ds, request.properties)

    # https://docs.ogc.org/DRAFTS/19-087.html#datetime-parameter-subset-requirements
    # requirement 7D: "If a datetime parameter is specified requesting a
    # coverage without any temporal dimension, the parameter SHALL either be
    # ignored, or a 4xx client error generated." We choose to ignore it.
    if request.datetime is not None and "time" in ds.variables:
        if isinstance(request.datetime, tuple):
            time_slice = slice(*request.datetime)
            time_slice = ensure_time_index_compatible(ds, time_slice)
            ds = ds.sel(time=time_slice)
        else:
            timespec = ensure_time_index_compatible(ds, request.datetime)
            ds = ds.sel(time=timespec, method="nearest").squeeze()

    if request.subset is not None:
        subset_bbox, ds = _apply_subsetting(ds, request.subset, subset_crs)
    else:
        subset_bbox = None

    if request.bbox is not None:
        ds = _apply_bbox(ds, request.bbox, bbox_crs, always_xy=False)

    # Do a provisional size check with an approximate scaling before attempting
    # to determine a grid mapping, so the client gets a comprehensible error
    # if the coverage is empty or too large.
    _assert_coverage_size_ok(CoverageScaling(request, final_crs, ds))

    transformed_gm = source_gm = GridMapping.from_dataset(ds, crs=native_crs)
    if native_crs != final_crs:
        transformed_gm = transformed_gm.transform(final_crs).to_regular()
    if transformed_gm is not source_gm:
        # We can't combine the scaling operation with this CRS transformation,
        # since the size may end up wrong after the re-application of bounding
        # boxes below.
        ds = resample_in_space(ds, source_gm=source_gm, target_gm=transformed_gm)

    if native_crs != final_crs:
        # If we've resampled into a new CRS, the transformed native-CRS
        # bbox[es] may have been too big, so we re-crop in the final CRS.
        if request.bbox is not None:
            ds = _apply_bbox(ds, request.bbox, bbox_crs, always_xy=False)
        if subset_bbox is not None:
            ds = _apply_bbox(ds, subset_bbox, subset_crs, always_xy=True)

    # Apply final size check and scaling operation after bbox, subsetting,
    # and CRS transformation, to make sure that the final size is correct.
    scaling = CoverageScaling(request, final_crs, ds)
    _assert_coverage_size_ok(scaling)
    if scaling.factor != (1, 1):
        cropped_gm = GridMapping.from_dataset(ds, crs=final_crs)
        scaled_gm = scaling.apply(cropped_gm)
        ds = resample_in_space(ds, source_gm=cropped_gm, target_gm=scaled_gm)

    ds.rio.write_crs(final_crs, inplace=True)
    for var in ds.data_vars.values():
        var.attrs.pop("grid_mapping", None)

    # check: rename axes to match final CRS?

    media_types = dict(
        tiff={"geotiff", "image/tiff", "application/x-geotiff"},
        png={"png", "image/png"},
        netcdf={"netcdf", "application/netcdf", "application/x-netcdf"},
    )
    if content_type in media_types["tiff"]:
        content = dataset_to_image(ds, "tiff", final_crs)
    elif content_type in media_types["png"]:
        content = dataset_to_image(ds, "png", final_crs)
    elif content_type in media_types["netcdf"]:
        content = dataset_to_netcdf(ds)
    else:
        # It's expected that the caller (server API handler) will catch
        # unhandled types, but we may as well do the right thing if any
        # do slip through.
        raise ApiError.UnsupportedMediaType(
            f"Unsupported media type {content_type}. "
            + "Available media types: "
            + ", ".join([type_ for value in media_types.values() for type_ in value])
        )
    final_bbox = get_bbox_from_ds(ds)
    if not is_xy_order(final_crs):
        final_bbox = final_bbox[1], final_bbox[0], final_bbox[3], final_bbox[2]
    return content, final_bbox, final_crs


def _apply_properties(collection_id, ds, properties):
    requested_vars = set(properties)
    data_vars = set(
        map(
            # Filter out 0-dimensional variables (usually grid mapping variables)
            str,
            {k: v for k, v in ds.data_vars.items() if v.dims != ()},
        )
    )
    unrecognized_vars = requested_vars - data_vars
    if unrecognized_vars == set():
        ds = ds.drop_vars(list(data_vars - requested_vars - {"crs", "spatial_ref"}))
    else:
        raise ApiError.BadRequest(
            f"The following properties are not present in the coverage "
            f'{collection_id}: {", ".join(unrecognized_vars)}'
        )
    return ds


def _assert_coverage_size_ok(scaling: CoverageScaling):
    size_limit = 4000 * 4000  # TODO make this configurable
    x, y = scaling.size
    if (x * y) > size_limit:
        raise ApiError.ContentTooLarge(
            f"Requested coverage is too large:" f"{x} × {y} > {size_limit}."
        )


_IndexerTuple = NamedTuple(
    "Indexers",
    [
        ("indices", dict[str, Any]),  # non-geographic single-valued specifiers
        ("slices", dict[str, slice]),  # non-geographic range specifiers
        ("x", Optional[Union[float, tuple[float, float]]]),  # x or longitude
        ("y", Optional[Union[float, tuple[float, float]]]),  # y or latitude
    ],
)


def _apply_subsetting(
    ds: xr.Dataset, subset_spec: dict, subset_crs: pyproj.CRS
) -> tuple[list[float], xr.Dataset]:
    indexers = _parse_subset_specifier(subset_spec, ds)

    # TODO: for geographic subsetting, also handle single-value (non-slice)
    #  indices and half-open slices.
    bbox = None
    if (indexers.x, indexers.y) != (None, None):
        bbox, ds = _apply_geographic_subsetting(ds, subset_crs, indexers)
    if indexers.slices:
        ds = ds.sel(indexers=indexers.slices)
    if indexers.indices:
        ds = ds.sel(indexers=indexers.indices, method="nearest")

    return bbox, ds


def _parse_subset_specifier(
    subset_spec: dict[str, Union[str, tuple[Optional[str], Optional[str]]]],
    ds: xr.Dataset,
) -> _IndexerTuple:
    specifiers = {}
    for axis, value in subset_spec.items():
        if isinstance(value, str):  # single value
            if axis == "time":
                specifiers[axis] = ensure_time_index_compatible(ds, value)
            else:
                try:
                    # Parse to float if possible
                    specifiers[axis] = float(value)
                except ValueError:
                    specifiers[axis] = value
        else:  # range
            low, high = value
            low = None if low == "*" else low
            high = None if high == "*" else high
            if axis == "time":
                specifiers[axis] = ensure_time_index_compatible(ds, slice(low, high))
            else:
                low = float(low)
                high = float(high)
                if axis.lower()[:3] in ["y", "n", "nor", "lat"] and high < low:
                    low, high = high, low
                specifiers[axis] = low, high

    # Find and extract geographic parameters, if any. These have to be
    # handled specially, since they refer to axis names in the subsetting
    # CRS rather than dimension names in the dataset. For now, we actually
    # don't check the CRS axis names, but just look for parameters with
    # appropriate names (x, y, lat, long, etc.).
    x_param, y_param = _find_geographic_parameters(list(specifiers))
    x_value = specifiers.pop(x_param) if x_param is not None else None
    y_value = specifiers.pop(y_param) if y_param is not None else None

    # Separate index and slice (i.e. single-value and range) specifiers
    indices = {k: v for k, v in specifiers.items() if not isinstance(v, slice)}
    slices = {k: v for k, v in specifiers.items() if isinstance(v, slice)}

    return _IndexerTuple(indices, slices, x_value, y_value)


def _apply_geographic_subsetting(
    ds, subset_crs, indexers
) -> tuple[list[float], xr.Dataset]:
    # NB: We use xy axis ordering for the bounding boxes throughout this
    # function, regardless of what's specified in the CRSs.

    # 1. transform native extent to a whole-dataset bbox in subset_crs.
    # We'll use this to fill in "full extent" values if geographic
    # subsetting is only specified in one dimension.
    full_bbox_native = get_bbox_from_ds(ds)
    native_crs = get_crs_from_dataset(ds)
    full_bbox_subset_crs = transform_bbox(full_bbox_native, native_crs, subset_crs)

    # 2. Find horizontal and/or vertical ranges in indexers, falling back to
    # values from whole-dataset bbox if a complete bbox is not specified.
    x0, x1 = (
        indexers.x
        if indexers.x is not None
        else (full_bbox_subset_crs[0], full_bbox_subset_crs[2])
    )
    y0, y1 = (
        indexers.y
        if indexers.y is not None
        else (full_bbox_subset_crs[1], full_bbox_subset_crs[3])
    )

    # 3. Using the ranges determined from the indexers and whole-dataset bbox,
    # construct the requested bbox in the subsetting CRS.
    bbox_subset_crs = [x0, y0, x1, y1]

    # 4. Transform requested bbox from subsetting CRS to dataset-native CRS.
    bbox_native_crs = transform_bbox(bbox_subset_crs, subset_crs, native_crs)

    # 6. Apply the dataset-native bbox using sel, making sure that y/latitude
    # slice has the same ordering as the corresponding co-ordinate.
    h_dim = get_h_dim(ds)
    v_dim = get_v_dim(ds)
    ds = ds.sel(
        indexers={
            h_dim: slice(bbox_native_crs[0], bbox_native_crs[2]),
            v_dim: slice(
                *_correct_inverted_y_range_if_necessary(
                    ds, v_dim, (bbox_native_crs[1], bbox_native_crs[3])
                )
            ),
        }
    )
    return bbox_subset_crs, ds


def get_bbox_from_ds(ds: xr.Dataset):
    h, v = ds[get_h_dim(ds)], ds[get_v_dim(ds)]
    bbox = list(map(float, [h[0], v[0], h[-1], v[-1]]))
    _ensure_bbox_y_ascending(bbox)
    return bbox


def _find_geographic_parameters(
    names: list[str],
) -> tuple[Optional[str], Optional[str]]:
    x, y = None, None
    for name in names:
        if name.lower()[:3] in ["x", "e", "eas", "lon"]:
            x = name
        if name.lower()[:3] in ["y", "n", "nor", "lat"]:
            y = name
    return x, y


def transform_bbox(
    bbox: list[float], source_crs: pyproj.CRS, dest_crs: pyproj.CRS
) -> list[float]:
    if source_crs == dest_crs:
        return bbox
    transformer = pyproj.Transformer.from_crs(source_crs, dest_crs, always_xy=True)
    bbox_ = bbox.copy()
    _ensure_bbox_y_ascending(bbox_)
    return list(transformer.transform_bounds(*bbox_))


def _apply_bbox(
    ds: xr.Dataset, bbox: list[float], bbox_crs: pyproj.CRS, always_xy: bool
):
    native_crs = get_crs_from_dataset(ds)

    if native_crs != bbox_crs:
        transformer = pyproj.Transformer.from_crs(
            bbox_crs, native_crs, always_xy=always_xy
        )
        _ensure_bbox_y_ascending(bbox, always_xy or is_xy_order(bbox_crs))
        bbox = transformer.transform_bounds(*bbox)
    h_dim = get_h_dim(ds)
    v_dim = get_v_dim(ds)
    x0, y0, x1, y1 = (
        (0, 1, 2, 3) if (always_xy or is_xy_order(native_crs)) else (1, 0, 3, 2)
    )
    v_slice = _correct_inverted_y_range_if_necessary(ds, v_dim, (bbox[y0], bbox[y1]))
    ds = ds.sel({h_dim: slice(bbox[x0], bbox[x1]), v_dim: slice(*v_slice)})
    return ds


def _ensure_bbox_y_ascending(bbox: list, xy_order: bool = True):
    y0, y1 = (1, 3) if xy_order else (0, 2)
    if bbox[y0] > bbox[y1]:
        bbox[y0], bbox[y1] = bbox[y1], bbox[y0]


def _correct_inverted_y_range_if_necessary(
    ds: xr.Dataset, axis: str, range_: tuple[float, float]
) -> tuple[float, float]:
    x0, x1 = range_
    # Make sure latitude slice direction matches axis direction.
    # (For longitude, a descending-order slice is valid.)
    if (
        None not in range_
        and axis[:3].lower() in {"lat", "nor", "y"}
        and (x0 < x1) != (ds[axis][0] < ds[axis][-1])
    ):
        x0, x1 = x1, x0
    return x0, x1


def dataset_to_image(
    ds: xr.Dataset,
    image_format: Literal["png", "tiff"] = "png",
    crs: pyproj.CRS = None,
) -> bytes:
    """Return an in-memory bitmap (TIFF or PNG) representing a dataset

    Args:
        ds: a dataset
        image_format: image format to generate ("png" or "tiff")
        crs: CRS of the dataset

    Returns:
        TIFF-formatted bytes representing the dataset
    """

    if image_format == "png":
        for var in ds.data_vars:
            # rasterio's PNG driver only supports these data types.
            if ds[var].dtype not in {np.uint8, np.uint16}:
                ds[var] = ds[var].astype(np.uint16, casting="unsafe")

    ds = ds.squeeze()

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "out." + image_format)
        # Make dataset representable in an image format by discarding
        # additional variables and dimensions.
        ds = ds.drop_vars(names=["crs", "spatial_ref"], errors="ignore").squeeze()
        if len(ds.data_vars) == 1:
            ds[list(ds.data_vars)[0]].rio.to_raster(path)
        else:
            ds.rio.to_raster(path)
        if crs is not None:
            with rasterio.open(path, mode="r+") as src:
                src.crs = crs
        with open(path, "rb") as fh:
            data = fh.read()
    return data


def dataset_to_netcdf(ds: xr.Dataset) -> bytes:
    """Return an in-memory NetCDF representing a dataset

    Args:
        ds: a dataset

    Returns:
        NetCDF-formatted bytes representing the dataset
    """
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "out.nc")
        ds.to_netcdf(path)
        with open(path, "rb") as fh:
            data = fh.read()
    return data


def get_coverage_domainset(ctx: DatasetsContext, collection_id: str):
    """Return the domain set of a dataset-backed coverage

    The domain set is the set of input parameters (e.g. geographical extent,
    time span) for which a coverage is defined.

    Args:
        ctx: a datasets context
        collection_id: the dataset for which to return the domain set

    Returns:
        a dictionary representing an OGC API - Coverages domain set
    """
    ds = get_dataset(ctx, collection_id)
    grid_limits = dict(
        type="GridLimits",
        srsName=f"http://www.opengis.net/def/crs/OGC/0/Index{len(ds.sizes)}D",
        axisLabels=list(ds.sizes),
        axis=[_get_grid_limits_axis(ds, dim) for dim in ds.sizes],
    )
    grid = dict(
        type="GeneralGridCoverage",
        srsName=get_crs_from_dataset(ds).to_string(),
        axisLabels=list(ds.sizes.keys()),
        axis=_get_axes_properties(ds),
        gridLimits=grid_limits,
    )
    return dict(type="DomainSet", generalGrid=grid)


def get_collection_metadata(ctx: DatasetsContext, collection_id: str):
    """Return a metadata dictionary for a dataset

    The metadata is taken directly from the dataset attributes.

    Args:
        ctx: a datasets context
        collection_id: the dataset for which to return the metadata

    Returns:
        a dictionary of metadata keys and values
    """
    ds = get_dataset(ctx, collection_id)
    return ds.attrs


def get_dataset(ctx: DatasetsContext, collection_id: str):
    """Get a dataset from a datasets context

    Args:
        ctx: a datasets context
        collection_id: the ID of a dataset in the context

    Returns:
        the dataset
    """
    ml_dataset = ctx.get_ml_dataset(collection_id)
    ds = ml_dataset.get_dataset(0)
    assert isinstance(ds, xr.Dataset)
    return ds


def _get_axes_properties(ds: xr.Dataset) -> list[dict]:
    return [_get_axis_properties(ds, dim) for dim in ds.sizes]


def _get_axis_properties(ds: xr.Dataset, dim: str) -> dict[str, Any]:
    axis = ds.coords[dim]
    if np.issubdtype(axis.dtype, np.datetime64):
        lower_bound = np.datetime_as_string(axis[0])
        upper_bound = np.datetime_as_string(axis[-1])
    else:
        lower_bound, upper_bound = axis[0].item(), axis[-1].item()
    return dict(
        type="RegularAxis",
        axisLabel=dim,
        lowerBound=lower_bound,
        upperBound=upper_bound,
        resolution=abs((axis[-1] - axis[0]).item() / len(axis)),
        uomLabel=get_units(ds, dim),
    )


def _get_grid_limits_axis(ds: xr.Dataset, dim: str) -> dict[str, Any]:
    return dict(type="IndexAxis", axisLabel=dim, lowerBound=0, upperBound=len(ds[dim]))


def get_units(ds: xr.Dataset, dim: str) -> str:
    coord = ds.coords[dim]
    if hasattr(coord, "attrs") and "units" in coord.attrs:
        return coord.attrs["units"]
    if np.issubdtype(coord, np.datetime64):
        return np.datetime_data(coord)[0]
    # TODO: as a fallback for spatial axes, we could try matching dimensions
    #  to CRS axes and take the unit from the CRS definition.
    return "unknown"


def get_crs_from_dataset(ds: xr.Dataset) -> pyproj.CRS:
    """Return the CRS of a dataset as a string. The CRS is taken from the
    metadata of the crs or spatial_ref variables, if available.
    "EPSG:4326" is used as a fallback.

    Args:
        ds: a dataset

    Returns:
        a string representation of the dataset's CRS, or "EPSG:4326" if
        the CRS cannot be determined
    """
    for var_name in "crs", "spatial_ref":
        if var_name in ds.variables:
            var = ds[var_name]
            for attr_name in "spatial_ref", "crs_wkt":
                if attr_name in var.attrs:
                    crs_string = ds[var_name].attrs[attr_name]
                    return pyproj.CRS(crs_string)
    return pyproj.CRS("EPSG:4326")


def get_coverage_rangetype(ctx: DatasetsContext, collection_id: str) -> dict[str, list]:
    """Return the range type of a dataset

    The range type describes the data types of the dataset's variables
    using a format defined in https://docs.ogc.org/is/09-146r6/09-146r6.html

    Args:
        ctx: datasets context
        collection_id: ID of the dataset in the supplied context

    Returns:
        a dictionary representing the specified dataset's range type
    """
    return get_coverage_rangetype_for_dataset(get_dataset(ctx, collection_id))


def get_coverage_rangetype_for_dataset(ds) -> dict[str, list]:
    """Return the range type of a dataset

    The range type describes the data types of the dataset's variables
    using a format defined in https://docs.ogc.org/is/09-146r6/09-146r6.html

    Args:
        ds: a dataset

    Returns:
        a dictionary representing the supplied dataset's range type
    """
    result = dict(type="DataRecord", field=[])
    for var_name, variable in ds.data_vars.items():
        if variable.dims == ():
            # A 0-dimensional variable is probably a grid mapping variable;
            # in any case, it doesn't have the dimensions of the cube, so
            # isn't part of the range.
            continue
        result["field"].append(
            dict(
                type="Quantity",
                name=var_name,
                description=get_dataarray_description(variable),
                encodingInfo=dict(dataType=dtype_to_opengis_datatype(variable.dtype)),
            )
        )
    return result


def dtype_to_opengis_datatype(dt: np.dtype) -> str:
    """Convert a NumPy dtype to an equivalent OpenGIS type identifier string.

    Args:
        dt: a NumPy dtype

    Returns:
        an equivalent OpenGIS type identifier string, or an empty string
        if the dtype is not recognized
    """
    nbits = 8 * np.dtype(dt).itemsize
    int_size_map = {8: "Byte", 16: "Short", 32: "Int", 64: "Long"}
    prefix = "http://www.opengis.net/def/dataType/OGC/0/"
    if np.issubdtype(dt, np.floating):
        opengis_type = f"{prefix}float{nbits}"
    elif np.issubdtype(dt, np.signedinteger):
        opengis_type = f"{prefix}signed{int_size_map[nbits]}"
    elif np.issubdtype(dt, np.unsignedinteger):
        opengis_type = f"{prefix}unsigned{int_size_map[nbits]}"
    elif "datetime64" in str(dt):
        opengis_type = "http://www.opengis.net/def/bipm/UTC"
    else:
        opengis_type = ""  # TODO decide what to do in this case
    return opengis_type


def get_dataarray_description(da: xr.DataArray) -> str:
    """Return a string describing a DataArray, either from an attribute or,
    as a fallback, from its name attribute.

    Args:
        da: a DataArray

    Returns:
        a string describing the DataArray
    """
    if hasattr(da, "attrs"):
        for attr in ["description", "long_name", "standard_name", "name"]:
            if attr in da.attrs:
                return da.attrs[attr]
    return str(da.name)


def get_collection_envelope(ds_ctx, collection_id):
    """Return the OGC API - Coverages envelope of a dataset.

    The envelope comprises the extents of all the dataset's dimensions.

    Args:
        ds_ctx: a datasets context
        collection_id: a dataset ID within the given context

    Returns:
        the envelope of the specified dataset
    """
    ds = get_dataset(ds_ctx, collection_id)
    return {
        "type": "EnvelopeByAxis",
        "srsName": get_crs_from_dataset(ds).to_string(),
        "axisLabels": list(ds.sizes.keys()),
        "axis": _get_axes_properties(ds),
    }


def is_xy_order(crs: pyproj.CRS) -> bool:
    """Try to determine whether a CRS has x-y axis order"""
    x_index = None
    y_index = None
    x_re = re.compile("^x|lon|east", flags=re.IGNORECASE)
    y_re = re.compile("^y|lat|north", flags=re.IGNORECASE)

    for i, axis in enumerate(crs.axis_info):
        for prop in "name", "abbrev", "direction":
            if x_re.search(getattr(axis, prop)):
                x_index = i
            elif y_re.search(getattr(axis, prop)):
                y_index = i

    if x_index is not None and y_index is not None:
        return x_index < y_index
    else:
        return True  # assume xy
