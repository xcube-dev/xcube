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
    scr_ij_bboxes, x_coords, y_coords = _get_scr_bboxes_indices(
        transformer, source_gm, target_gm
    )

    # transform grid points from traget grid mapping to source grid mapping
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
        slices_reprojected = []
        for idx, chunk_size in enumerate(data_array.chunks[0]):
            # reorganize data array slice to align with the
            # chunks of source_xx and source_yy
            scr_data = _reorganize_data_array_slice(
                data_array,
                x_coords,
                y_coords,
                idx,
                chunk_size,
                scr_ij_bboxes,
                fill_value,
            )

            data_reprojected = da.map_blocks(
                _reproject_block,
                source_xx,
                source_yy,
                scr_data,
                x_coords,
                y_coords,
                dtype=data_array.dtype,
                chunks=(
                    scr_data.shape[0],
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
        ix = np.rint(ix).astype(np.int16)
        iy = np.rint(iy).astype(np.int16)
        data_reprojected = scr_data[:, iy, ix]
    else:
        raise NotImplementedError()

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

    return scr_ij_bboxes, x_coords, y_coords


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
    idx: int,
    chunk_size_dim0: int,
    scr_ij_bboxes: np.ndarray,
    fill_value: int | float | None,
) -> da.Array:
    fill_value_da = _get_fill_value(data_array, fill_value)

    dim0_start = idx * chunk_size_dim0
    dim0_end = (idx + 1) * chunk_size_dim0

    scr_data = da.zeros(
        (
            dim0_end - dim0_start,
            y_coords.shape[0] * scr_ij_bboxes.shape[1],
            x_coords.shape[0] * scr_ij_bboxes.shape[2],
        ),
        chunks=(-1, y_coords.shape[0], x_coords.shape[0]),
        dtype=data_array.dtype,
    )

    for i in range(scr_ij_bboxes.shape[2]):
        for j in range(scr_ij_bboxes.shape[1]):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]
            if _target_chunk_fully_intersects_source(scr_ij_bbox, data_array):
                scr_data[
                    :,
                    j * y_coords.shape[0] : (j + 1) * y_coords.shape[0],
                    i * x_coords.shape[0] : (i + 1) * x_coords.shape[0],
                ] = data_array.data[
                    dim0_start:dim0_end,
                    scr_ij_bbox[1] : scr_ij_bbox[3],
                    scr_ij_bbox[0] : scr_ij_bbox[2],
                ]
            elif _target_chunk_not_intersects_source(scr_ij_bbox, data_array):
                scr_data[
                    :,
                    j * y_coords.shape[0] : (j + 1) * y_coords.shape[0],
                    i * x_coords.shape[0] : (i + 1) * x_coords.shape[0],
                ] = da.full(
                    (
                        dim0_end - dim0_start,
                        y_coords.shape[0],
                        x_coords.shape[0],
                    ),
                    fill_value_da,
                    chunks=(-1, -1, -1),
                    dtype=data_array.dtype,
                )
            else:
                scr_ij_bbox_mod = [
                    max(0, int(scr_ij_bbox[0])),
                    max(0, int(scr_ij_bbox[1])),
                    min(data_array.shape[2], int(scr_ij_bbox[2])),
                    min(data_array.shape[1], int(scr_ij_bbox[3])),
                ]
                data = data_array.data[
                    dim0_start:dim0_end,
                    scr_ij_bbox_mod[1] : scr_ij_bbox_mod[3],
                    scr_ij_bbox_mod[0] : scr_ij_bbox_mod[2],
                ]
                pad_width = (
                    (0, 0),
                    (
                        -min(0, int(scr_ij_bbox[1])),
                        max(0, int(scr_ij_bbox[3] - data_array.shape[1])),
                    ),
                    (
                        -min(0, int(scr_ij_bbox[0])),
                        max(0, int(scr_ij_bbox[2] - data_array.shape[2])),
                    ),
                )
                scr_data[
                    :,
                    j * y_coords.shape[0] : (j + 1) * y_coords.shape[0],
                    i * x_coords.shape[0] : (i + 1) * x_coords.shape[0],
                ] = da.pad(
                    data, pad_width, mode="constant", constant_values=fill_value_da
                )

    return scr_data


def _target_chunk_not_intersects_source(
    scr_ij_bbox: np.ndarray, data_array: xr.DataArray
):
    return (
        scr_ij_bbox[2] < 0
        or scr_ij_bbox[0] >= data_array.data.shape[2]
        or scr_ij_bbox[3] < 0
        or scr_ij_bbox[1] >= data_array.data.shape[1]
    )


def _target_chunk_fully_intersects_source(
    scr_ij_bbox: np.ndarray, data_array: xr.DataArray
):
    return (
        scr_ij_bbox[0] >= 0
        and scr_ij_bbox[2] < data_array.data.shape[2]
        and scr_ij_bbox[1] >= 0
        and scr_ij_bbox[3] < data_array.data.shape[1]
    )
