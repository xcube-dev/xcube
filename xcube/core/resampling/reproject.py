# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math
from collections.abc import Hashable, Mapping
from typing import Any

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.core.geom import clip_dataset_by_geometry
from xcube.core.gridmapping import GridMapping

from .affine import affine_transform_dataset

_FILLVALUE_UINT8 = 255
_FILLVALUE_UINT16 = 65535
_FILLVALUE_INT = -1
_FILLVALUE_FLOAT = np.nan
_INTERPOLATIONS = {"nearest": 0, "triangular": 1, "bilinear": 2}
_SCALE_LIMIT = 0.95  # scale limit to decide with down-scaling is applied


def reproject_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping | None = None,
    target_gm: GridMapping | None = None,
    ref_ds: xr.Dataset | None = None,
    fill_value: int | float | None = None,
    interpolation: str | None = "nearest",
    downscale_var_configs: Mapping[Hashable, Mapping[str, Any]] | None = None,
):
    """
    Reprojects a dataset to a new coordinate reference system.

    This function reprojects a dataset (`source_ds`) to a target projection defined
    by `target_gm`, or inferred from a reference dataset (`ref_ds`) if `target_gm`
    is not provided.

    The input dataset must have two spatial coordinates. The function also
    supports 3D data cubes, as long as the non-spatial dimension is the first
    (e.g., variables shaped like ("time", "y", "x")). Any data variable that does not
    have two spatial coordinates in the last two dimensions will be excluded from
    the output.

    Args:
        source_ds: The dataset to reproject.
        source_gm: Optional. Grid mapping associated with the source dataset.If not
            given, `source_gm` is derived from `source_ds`.
        target_gm: Optional. Target grid mapping. Required if `ref_ds` is not provided.
        ref_ds: Optional. A reference dataset used to derive the target grid mapping.
        fill_value: Optional. Fill value for areas outside input bounds. If not
            provided, defaults are chosen based on data type:
                - float: NaN
                - uint8: 255
                - uint16: 65535
                - other integers: -1
        interpolation: Optional. Interpolation method to use. Must be one of:
            "nearest", "triangular", or "bilinear". Defaults to "nearest".
            "triangular" uses 3 adjacent pixels, "bilinear" uses 4. These methods
            apply only to floating-point data. Convert integer data to float if
            interpolation is needed.
        downscale_var_configs (Optional[Dict[str, Dict]]): Optional resampling
            configurations for individual variables. Used when the target resolution
            is coarser than the source. It is a mapping from variable names to
            configuration dictionaries which can have the following properties:
               - ``aggregator`` (str) - An optional aggregating
                    function. It is used for down-sampling only.
                    Examples are ``numpy.nanmean``, ``numpy.nanmin``,
                    ``numpy.nanmax``.
                    Default is ``numpy.nanmean`` for floating point variables,
                    and None (= nearest neighbor) for integer and bool variables.
               - ``recover_nan`` (bool) - whether a special algorithm
                    shall be used that is able to recover values that would
                    otherwise yield NaN during resampling.
                    Default is False for all variable types since this
                    may require considerable CPU resources on top.

    Returns:
        The reprojected dataset includes only those variables whose last two
        dimensions are spatial coordinates. The grid mapping information is
        stored as a coordinate named `spatial_ref`, which is the default convention
        in `xarray`.


    Notes:
        This method is a high-performance alternative to `xcube.core.resampling.resample_in_space`
        for reprojecting datasets to different coordinate reference systems (CRS). It is
        ideal for reprojection between regular grids. It improves computational efficiency
        and simplifies the reprojection process.

        The methods `reproject_dataset` and `resample_in_space` produce nearly identical
        results when reprojecting to a different CRS, with only negligible differences.
        `resample_in_space` remains available to preserve compatibility with existing
        services. Once `reproject_dataset` proves stable in production use, it may be
        integrated into `resample_in_space`.
    """

    # translate interpolation mode
    interpolation_mode = _INTERPOLATIONS.get(interpolation or "nearest")
    if not isinstance(interpolation_mode, int):
        raise ValueError(
            f"invalid interpolation: {interpolation!r}. The argument "
            "interpolation must be one of 'nearest', 'triangular', or 'bilinear'."
        )

    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None and ref_ds is not None:
        target_gm = GridMapping.from_dataset(ref_ds)
    elif ref_ds is None and target_gm is None:
        raise ValueError("Either ref_ds or target_gm needs to be given.")

    transformer = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )

    # If source has higher resolution than target, downscale first, then reproject
    source_ds, source_gm = _downscale_source_dataset(
        source_ds, source_gm, target_gm, transformer, var_configs=downscale_var_configs
    )

    # For each bounding box in the target grid mapping:
    # - determine the indices of the bbox in the source dataset
    # - extract the corresponding coordinates for each bbox in the source dataset
    # - compute the pad_width to handle areas requested by target_gm that exceed the
    #   bounds of source_gm.
    scr_ij_bboxes, x_coords, y_coords, pad_width = _get_scr_bboxes_indices(
        transformer, source_gm, target_gm
    )

    # transform grid points from target grid mapping to source grid mapping
    source_xx, source_yy = _transform_gridpoints(transformer, target_gm)

    # reproject dataset
    x_name, y_name = source_gm.xy_dim_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name))
    x_name, y_name = target_gm.xy_dim_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    ds_out = xr.Dataset(
        coords=coords,
        attrs=source_ds.attrs,
    )
    xy_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.items():
        if data_array.dims[-2:] != xy_dims:
            continue

        # treat 2d arrays as 3d arrays
        assert len(data_array.dims) in (
            2,
            3,
        ), f"Data variable {var_name} has {len(data_array.dims)} dimensions."
        data_array_expanded = False
        if len(data_array.dims) == 2:
            data_array = data_array.expand_dims({"dummy": 1})
            data_array_expanded = True

        numpy_array = False
        if isinstance(data_array.data, np.ndarray):
            numpy_array = True
            data_array = data_array.copy(data=da.from_array(data_array.data, chunks={}))

        # reorganize data array slice to align with the
        # chunks of source_xx and source_yy
        scr_data = _reorganize_data_array_slice(
            data_array,
            x_coords,
            y_coords,
            scr_ij_bboxes,
            pad_width,
            fill_value,
        )
        slices_reprojected = []
        # calculate reprojection of each chunk along the 1st (non-spatial) dimension.
        for idx, chunk_size in enumerate(data_array.chunks[0]):
            dim0_start = idx * chunk_size
            dim0_end = (idx + 1) * chunk_size

            data_reprojected = da.map_blocks(
                _reproject_block,
                source_xx,
                source_yy,
                scr_data[dim0_start:dim0_end],
                x_coords,
                y_coords,
                dtype=data_array.dtype,
                chunks=(
                    scr_data[dim0_start:dim0_end].shape[0],
                    source_yy.chunks[0][0],
                    source_yy.chunks[1][0],
                ),
                scr_x_res=source_gm.x_res,
                scr_y_res=source_gm.y_res,
                interpolation_mode=interpolation_mode,
            )
            data_reprojected = data_reprojected[
                :, : target_gm.height, : target_gm.width
            ]
            slices_reprojected.append(data_reprojected)
        array_reprojected = da.concatenate(slices_reprojected, axis=0)
        if numpy_array:
            array_reprojected = array_reprojected.compute()
        if data_array_expanded:
            ds_out[var_name] = (
                (y_name, x_name),
                array_reprojected[0, :, :],
            )
        else:
            ds_out[var_name] = (
                (data_array.dims[0], y_name, x_name),
                array_reprojected,
            )
        ds_out[var_name].attrs = data_array.attrs
    return ds_out


def _reproject_block(
    source_xx: np.ndarray,
    source_yy: np.ndarray,
    scr_data: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    scr_x_res: int | float,
    scr_y_res: int | float,
    interpolation_mode: int,
) -> np.ndarray:
    ix = (source_xx - x_coord[0]) / scr_x_res
    iy = (source_yy - y_coord[0]) / -scr_y_res

    if interpolation_mode == 0:
        # interpolation == "nearest"
        ix = np.rint(ix).astype(np.int16)
        iy = np.rint(iy).astype(np.int16)
        data_reprojected = scr_data[:, iy, ix]
    elif interpolation_mode == 1:
        # interpolation == "triangular"
        ix_ceil = np.ceil(ix).astype(np.int16)
        ix_floor = np.floor(ix).astype(np.int16)
        iy_ceil = np.ceil(iy).astype(np.int16)
        iy_floor = np.floor(iy).astype(np.int16)
        diff_ix = ix - ix_floor
        diff_iy = iy - iy_floor
        value_00 = scr_data[:, iy_floor, ix_floor]
        value_01 = scr_data[:, iy_floor, ix_ceil]
        value_10 = scr_data[:, iy_ceil, ix_floor]
        value_11 = scr_data[:, iy_ceil, ix_ceil]
        mask = diff_ix + diff_iy < 1.0
        mask_3d = np.repeat(mask[np.newaxis, :, :], scr_data.shape[0], axis=0)
        diff_ix = np.repeat(diff_ix[np.newaxis, :, :], scr_data.shape[0], axis=0)
        diff_iy = np.repeat(diff_iy[np.newaxis, :, :], scr_data.shape[0], axis=0)
        data_reprojected = np.zeros(
            (scr_data.shape[0], iy.shape[0], iy.shape[1]), dtype=scr_data.dtype
        )
        # Closest triangle
        data_reprojected[mask_3d] = (
            value_00[mask_3d]
            + diff_ix[mask_3d] * (value_01[mask_3d] - value_00[mask_3d])
            + diff_iy[mask_3d] * (value_10[mask_3d] - value_00[mask_3d])
        )
        # Opposite triangle
        data_reprojected[~mask_3d] = (
            value_11[~mask_3d]
            + (1.0 - diff_ix[~mask_3d]) * (value_10[~mask_3d] - value_11[~mask_3d])
            + (1.0 - diff_iy[~mask_3d]) * (value_01[~mask_3d] - value_11[~mask_3d])
        )
    else:
        # interpolation == "bilinear"
        ix_ceil = np.ceil(ix).astype(np.int16)
        ix_floor = np.floor(ix).astype(np.int16)
        iy_ceil = np.ceil(iy).astype(np.int16)
        iy_floor = np.floor(iy).astype(np.int16)
        diff_ix = ix - ix_floor
        diff_iy = iy - iy_floor
        value_00 = scr_data[:, iy_floor, ix_floor]
        value_01 = scr_data[:, iy_floor, ix_ceil]
        value_10 = scr_data[:, iy_ceil, ix_floor]
        value_11 = scr_data[:, iy_ceil, ix_ceil]
        value_u0 = value_00 + diff_ix * (value_01 - value_00)
        value_u1 = value_10 + diff_ix * (value_11 - value_10)
        data_reprojected = value_u0 + diff_iy * (value_u1 - value_u0)

    return data_reprojected


def _downscale_source_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    transformer: pyproj.Transformer,
    var_configs: Mapping[Hashable, Mapping[str, Any]] | None = None,
):
    bbox_trans = transformer.transform_bounds(*target_gm.xy_bbox)
    xres_trans = (bbox_trans[2] - bbox_trans[0]) / target_gm.width
    yres_trans = (bbox_trans[3] - bbox_trans[1]) / target_gm.height
    x_scale = source_gm.x_res / xres_trans
    y_scale = source_gm.y_res / yres_trans
    if x_scale < _SCALE_LIMIT or y_scale < _SCALE_LIMIT:
        # clip source dataset to the transformed bounding box defined by
        # target grid mapping, so that affine_transform_dataset is not that heavy
        bbox_trans = [
            bbox_trans[0] - 2 * source_gm.x_res,
            bbox_trans[1] - 2 * source_gm.y_res,
            bbox_trans[2] + 2 * source_gm.x_res,
            bbox_trans[3] + 2 * source_gm.y_res,
        ]
        source_ds = clip_dataset_by_geometry(source_ds, bbox_trans)
        source_gm = GridMapping.from_dataset(source_ds)
        w, h = round(x_scale * source_gm.width), round(y_scale * source_gm.height)
        downscaled_size = (w if w >= 2 else 2, h if h >= 2 else 2)
        downscale_target_gm = GridMapping.regular(
            size=downscaled_size,
            xy_min=(source_gm.xy_bbox[0], source_gm.xy_bbox[1]),
            xy_res=(xres_trans, yres_trans),
            crs=source_gm.crs,
            tile_size=source_gm.tile_size,
        )
        source_ds = affine_transform_dataset(
            source_ds, source_gm, downscale_target_gm, var_configs
        )
        var_bounds = [
            source_ds[var_name].attrs["bounds"]
            for var_name in source_gm.xy_var_names
            if "bounds" in source_ds[var_name].attrs
        ]
        source_ds = source_ds.drop_vars(var_bounds)
        source_gm = GridMapping.from_dataset(source_ds)

    return source_ds, source_gm


def _get_scr_bboxes_indices(
    transformer: pyproj.Transformer,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> (np.ndarray, da.Array, da.Array, tuple[tuple[int]]):
    num_tiles_x = math.ceil(target_gm.width / target_gm.tile_width)
    num_tiles_y = math.ceil(target_gm.height / target_gm.tile_height)

    # get bboxes indices in source grid mapping
    origin = source_gm.x_coords.values[0], source_gm.y_coords.values[0]
    scr_ij_bboxes = np.full((4, num_tiles_y, num_tiles_x), -1, dtype=np.int32)
    for idx, xy_bbox in enumerate(target_gm.xy_bboxes):
        j, i = np.unravel_index(idx, (num_tiles_y, num_tiles_x))
        source_xy_bbox = transformer.transform_bounds(*xy_bbox)
        i_min = math.floor((source_xy_bbox[0] - origin[0]) / source_gm.x_res)
        i_max = math.ceil((source_xy_bbox[2] - origin[0]) / source_gm.x_res)
        if source_gm.is_j_axis_up:
            j_min = math.floor((source_xy_bbox[1] - origin[1]) / source_gm.y_res)
            j_max = math.ceil((source_xy_bbox[3] - origin[1]) / source_gm.y_res)
        else:
            j_min = math.floor((origin[1] - source_xy_bbox[3]) / source_gm.y_res)
            j_max = math.ceil((origin[1] - source_xy_bbox[1]) / source_gm.y_res)
        scr_ij_bboxes[:, j, i] = [i_min, j_min, i_max, j_max]

    # Extend bounding box indices to match the largest bounding box.
    # This ensures uniform chunk sizes, which are required for da.map_blocks.
    i_diff = scr_ij_bboxes[2] - scr_ij_bboxes[0]
    j_diff = scr_ij_bboxes[3] - scr_ij_bboxes[1]
    i_diff_max = np.max(i_diff) + 1
    j_diff_max = np.max(j_diff) + 1
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]

            i_half = (i_diff_max - i_diff[j, i]) // 2
            i_start = scr_ij_bbox[0] - i_half
            i_end = i_start + i_diff_max

            j_half = (j_diff_max - j_diff[j, i]) // 2
            j_start = scr_ij_bbox[1] - j_half
            j_end = j_start + j_diff_max

            scr_ij_bboxes[:, j, i] = [i_start, j_start, i_end, j_end]

    # gather the coordinates; coordinates will be extended
    # if they are outside the source grid mapping
    x_coords = np.zeros((i_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    y_coords = np.zeros((j_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    i_min = np.min(scr_ij_bboxes[0])
    i_max = np.max(scr_ij_bboxes[2])
    j_min = np.min(scr_ij_bboxes[[1, 3]])
    j_max = np.max(scr_ij_bboxes[[1, 3]])
    x_start = source_gm.x_coords.values[0] + i_min * source_gm.x_res
    x_end = source_gm.x_coords.values[0] + i_max * source_gm.x_res
    x_coord = np.arange(x_start, x_end, source_gm.x_res)
    y_res = source_gm.y_coords.values[1] - source_gm.y_coords.values[0]
    y_start = source_gm.y_coords.values[0] + j_min * y_res
    y_end = source_gm.y_coords.values[0] + j_max * y_res
    y_coord = np.arange(y_start, y_end, y_res)
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]

            i_start = scr_ij_bbox[0] - i_min
            i_end = i_start + i_diff_max
            x_coords[:, j, i] = x_coord[i_start:i_end]

            j_start = scr_ij_bbox[1] - j_min
            j_end = j_start + j_diff_max
            y_coords[:, j, i] = y_coord[j_start:j_end]

    x_coords = da.from_array(x_coords, chunks=(-1, 1, 1))
    y_coords = da.from_array(y_coords, chunks=(-1, 1, 1))

    pad_width = (
        (0, 0),
        (
            -min(0, int(j_min)),
            max(0, int(j_max - source_gm.height)),
        ),
        (
            -min(0, int(i_min)),
            max(0, int(i_max - source_gm.width)),
        ),
    )
    scr_ij_bboxes[[1, 3]] += pad_width[1][0]
    scr_ij_bboxes[[0, 2]] += pad_width[2][0]

    return scr_ij_bboxes, x_coords, y_coords, pad_width


def _transform_gridpoints(
    transformer: pyproj.Transformer, target_gm: GridMapping
) -> (da.Array, da.Array):
    # get meshed coordinates
    target_x = da.from_array(target_gm.x_coords.values, chunks=target_gm.tile_width)
    target_y = da.from_array(target_gm.y_coords.values, chunks=target_gm.tile_height)
    target_xx, target_yy = da.meshgrid(target_x, target_y)

    # get transformed coordinates
    # noinspection PyShadowingNames
    def transform_block(target_xx: np.ndarray, target_yy: np.ndarray):
        trans_xx, trans_yy = transformer.transform(target_xx, target_yy)
        return np.stack([trans_xx, trans_yy])

    source_xx_yy = da.map_blocks(
        transform_block,
        target_xx,
        target_yy,
        dtype=np.float32,
        chunks=(2, target_yy.chunks[0][0], target_yy.chunks[1][0]),
    )
    source_xx = source_xx_yy[0]
    source_yy = source_xx_yy[1]

    return source_xx, source_yy


def _get_fill_value(
    data_array: xr.DataArray, fill_value: int | float | None
) -> int | float:
    if fill_value is None:
        if data_array.dtype == np.uint8:
            fill_value = _FILLVALUE_UINT8
        elif data_array.dtype == np.uint16:
            fill_value = _FILLVALUE_UINT16
        elif np.issubdtype(data_array.dtype, np.integer):
            fill_value = _FILLVALUE_INT
        else:
            fill_value = _FILLVALUE_FLOAT

    return fill_value


def _reorganize_data_array_slice(
    data_array: xr.DataArray,
    x_coords: da.Array,
    y_coords: da.Array,
    scr_ij_bboxes: np.ndarray,
    pad_width: tuple[tuple[int]],
    fill_value: int | float | None,
) -> da.Array:
    fill_value_da = _get_fill_value(data_array, fill_value)

    data_out = da.zeros(
        (
            data_array.data.shape[0],
            y_coords.shape[0] * scr_ij_bboxes.shape[1],
            x_coords.shape[0] * scr_ij_bboxes.shape[2],
        ),
        chunks=(data_array.chunks[0][0], y_coords.shape[0], x_coords.shape[0]),
        dtype=data_array.dtype,
    )
    data_in = da.pad(
        data_array.data, pad_width, mode="constant", constant_values=fill_value_da
    )
    for i in range(scr_ij_bboxes.shape[2]):
        for j in range(scr_ij_bboxes.shape[1]):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]
            data_out[
                :,
                j * y_coords.shape[0] : (j + 1) * y_coords.shape[0],
                i * x_coords.shape[0] : (i + 1) * x_coords.shape[0],
            ] = data_in[
                :,
                scr_ij_bbox[1] : scr_ij_bbox[3],
                scr_ij_bbox[0] : scr_ij_bbox[2],
            ]

    return data_out
