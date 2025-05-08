# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import math

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping

FILLVALUE_UINT8 = 255
FILLVALUE_UINT16 = 65535
FILLVALUE_INT = -1
FILLVALUE_FLOAT = np.nan


def reproject_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping | None = None,
    target_gm: GridMapping | None = None,
    ref_ds: xr.Dataset | None = None,
    fill_value: int | float | None = None,
    interpolation: int = 0,
):
    """This function reprojects a dataset *source_ds*
    to another projection defined by *target_gm* or derived by *ref_ds*.

    The function expects *source_ds* or the given
    *source_gm* to have a two-dimensional
    coordinate variables that provide spatial x,y coordinates
    for every data variable with the same spatial dimensions.

    For example, a dataset may comprise variables with
    spatial dimensions ``var(..., y_dim, x_dim)``,
    then the function expects coordinates to be provided
    in two forms:

    2. Two-dimensional ``x_var(y_dim, x_dim)``
       and ``y_var(y_dim, x_dim)`` (coordinate) variables.

    If *target_gm* is given, and it defines a tile size,
    or *tile_size* is given and the number of tiles is
    greater than one in the output's x- or y-direction, then the
    returned dataset will be composed of lazy, chunked dask
    arrays. Otherwise, the returned dataset will be composed
    of ordinary numpy arrays.

    Args:
        source_ds: Source dataset.
        source_gm: Source dataset grid mapping.
        target_gm: Optional target geometry. If not given, output
            geometry will be computed to spatially fit *dataset* and to
            retain its spatial resolution.
        ref_ds: An optional dataset that provides the
            target grid mapping if *target_gm* is not provided.
            If *ref_ds* is given, its coordinate variables are copied
            by reference into the returned dataset.
        fill_value: fill value to be used for exceeding edges; if None, default
            values are taken defined by the data type. Fill value for float datasets
            is set to `np.nan`, fill value for uint8 is set to 255, fill value for
            uint16 is set to 65535, for all remaining integer datasets fill value
            is set to -1.
        interpolation: Interpolation method for computing output pixels.
            If given, must be "nearest", "triangular", or "bilinear".
            The default is "nearest". The "triangular" interpolation is
            performed between 3 and "bilinear" between 4 adjacent source
            pixels. Both are applied only to variables of
            floating point type. If you need to interpolate between
            integer data you should cast it to float first.

    Returns:
        A reprojected dataset, or None if the requested output does not
        intersect with *dataset*.
    """
    # TODO: treat 2d and 3d case
    # TODO: select interpolation based on dtype
    # TODO: test interpolation
    # TODO: write tests
    # TODO: write comments for code understadning
    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)

    if target_gm is None and ref_ds is not None:
        target_gm = GridMapping.from_dataset(ref_ds)
    elif ref_ds is None and target_gm is None:
        raise ValueError("Either ref_ds or target_gm needs to be given.")

    transformer = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )

    # get indices for each box in source dataset
    scr_ij_bboxes, x_coords, y_coords, pad_width = _get_scr_bboxes_indices(
        transformer, source_gm, target_gm
    )

    # transform grid points from target grid mapping to source grid mapping
    source_xx, source_yy = _transform_gridpoints(transformer, target_gm)

    # reproject dataset
    x_name, y_name = target_gm.xy_dim_names
    ds_out = xr.Dataset(
        coords={
            "time": source_ds.time,
            x_name: target_gm.x_coords,
            y_name: target_gm.y_coords,
            "spatial_ref": xr.DataArray(0, attrs=target_gm.crs.to_cf()),
        },
        attrs=source_ds.attrs,
    )
    xy_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.items():
        if data_array.dims[-2:] != xy_dims:
            continue
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
                interpolation=interpolation,
            )
            data_reprojected = data_reprojected[
                :, : target_gm.height, : target_gm.width
            ]
            slices_reprojected.append(data_reprojected)
        ds_out[var_name] = (
            ("time", y_name, x_name),
            da.concatenate(slices_reprojected, axis=0),
        )
    return ds_out


def _reproject_block(
    source_xx: np.ndarray,
    source_yy: np.ndarray,
    scr_data: np.ndarray,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    scr_x_res: int | float,
    scr_y_res: int | float,
    interpolation: int,
):
    ix = (source_xx - x_coord[0]) / scr_x_res
    iy = (source_yy - y_coord[0]) / -scr_y_res

    if interpolation == 0:
        # interpolation == "nearest"
        ix = np.rint(ix).astype(np.int16)
        iy = np.rint(iy).astype(np.int16)
        data_reprojected = scr_data[:, iy, ix]
    elif interpolation == 1:
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
        mask = np.repeat(mask[np.newaxis, :, :], scr_data.shape[0], axis=0)
        data_reprojected = np.zeros(
            (scr_data.shape[0], iy.shape[0], iy.shape[1]), dtype=scr_data.dtype
        )
        # Closest triangle
        data_reprojected[mask] = (
            value_00[mask]
            + diff_ix[mask] * (value_01[mask] - value_00[mask])
            + diff_iy[mask] * (value_10[mask] - value_00[mask])
        )
        # Opposite triangle
        data_reprojected[~mask] = (
            value_11[~mask]
            + (1.0 - diff_ix[~mask]) * (value_10[~mask] - value_11[~mask])
            + (1.0 - diff_iy[~mask]) * (value_01[~mask] - value_11[~mask])
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


def _get_scr_bboxes_indices(
    transformer: pyproj.Transformer,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> (np.ndarray, da.Array, da.Array):
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
    i_diff_max = np.max(i_diff)
    j_diff_max = np.max(j_diff)
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
            fill_value = FILLVALUE_UINT8
        elif data_array.dtype == np.uint16:
            fill_value = FILLVALUE_UINT16
        elif np.issubdtype(data_array.dtype, np.integer):
            fill_value = FILLVALUE_INT
        else:
            fill_value = FILLVALUE_FLOAT

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
