import math

# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import dask.array as da
import numpy as np
import pyproj
import xarray as xr

from xcube.core.gridmapping import GridMapping


def reproject_dataset(
    ds: xr.Dataset,
    target_gm: GridMapping,
    fill_value: int | float = -1,
    interpolation: int = 0,
):
    # get ij_bbox in source dataset
    source_gm = GridMapping.from_dataset(ds)
    trans_backward = pyproj.Transformer.from_crs(
        target_gm.crs, source_gm.crs, always_xy=True
    )
    num_tiles_x = math.ceil(target_gm.width / target_gm.tile_width)
    num_tiles_y = math.ceil(target_gm.height / target_gm.tile_height)
    scr_ij_bboxes = np.full((4, num_tiles_y, num_tiles_x), -1, dtype=np.int32)
    for idx, xy_bbox in enumerate(target_gm.xy_bboxes):
        idx_y, idx_x = np.unravel_index(idx, (num_tiles_y, num_tiles_x))
        source_xy_bbox = trans_backward.transform_bounds(*xy_bbox)
        if (
            source_xy_bbox[2] < source_gm.xy_bbox[0]
            or source_xy_bbox[0] > source_gm.xy_bbox[2]
            or source_xy_bbox[3] < source_gm.xy_bbox[1]
            or source_xy_bbox[1] > source_gm.xy_bbox[3]
        ):
            continue

        j_min = np.argmin(abs(source_gm.x_coords.values - source_xy_bbox[0]))
        j_max = np.argmin(abs(source_gm.x_coords.values - source_xy_bbox[2]))
        if j_min != 0:
            j_min -= 1
        if j_max != source_gm.width:
            j_max += 1
        i_min = np.argmin(abs(source_gm.y_coords.values - source_xy_bbox[1]))
        i_max = np.argmin(abs(source_gm.y_coords.values - source_xy_bbox[3]))
        if i_min > i_max:
            tmp0, tmp1 = i_min, i_max
            i_min, i_max = tmp1, tmp0
        if i_min != 0:
            i_min -= 1
        if i_max != source_gm.height:
            i_max += 1
        scr_ij_bboxes[:, idx_y, idx_x] = [j_min, i_min, j_max, i_max]

    # get largest bbox
    i_diff = scr_ij_bboxes[2] - scr_ij_bboxes[0]
    j_diff = scr_ij_bboxes[3] - scr_ij_bboxes[1]
    i_diff_max = np.max(i_diff)
    j_diff_max = np.max(j_diff)
    x_coords = np.zeros((i_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    y_coords = np.zeros((j_diff_max, num_tiles_y, num_tiles_x), dtype=np.float32)
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]

            i_half = (i_diff_max - i_diff[j, i]) // 2
            i_start = scr_ij_bbox[0] - i_half
            i_end = i_start + i_diff_max
            x_coords[:, j, i] = source_gm.x_coords.data[i_start:i_end]

            j_half = (j_diff_max - j_diff[j, i]) // 2
            j_start = scr_ij_bbox[1] - j_half
            j_end = j_start + j_diff_max
            y_coords[:, j, i] = source_gm.y_coords.data[j_start:j_end]

            scr_ij_bboxes[:, j, i] = [i_start, j_start, i_end, j_end]
    x_coords = da.from_array(x_coords, chunks=(-1, 1, 1))
    y_coords = da.from_array(y_coords, chunks=(-1, 1, 1))

    # get meshed coordinates
    target_x = da.from_array(target_gm.x_coords.values, chunks=target_gm.tile_width)
    target_y = da.from_array(target_gm.y_coords.values, chunks=target_gm.tile_height)
    target_xx, target_yy = da.meshgrid(target_x, target_y)

    # get transformed coordinates
    def transform_block(target_xx: np.ndarray, target_yy: np.ndarray):
        trans_xx, trans_yy = trans_backward.transform(target_xx, target_yy)
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

    # reproject dataset
    x_name, y_name = target_gm.xy_dim_names
    ds_out = xr.Dataset(
        coords={
            "time": ds.time,
            x_name: target_gm.x_coords,
            y_name: target_gm.y_coords,
            "spatial_ref": xr.DataArray(0, attrs=target_gm.crs.to_cf()),
        },
        attrs=ds.attrs,
    )
    xy_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in ds.items():
        if data_array.dims[-2:] != xy_dims:
            continue
        slices_reprojected = []
        for idx, chunk_size in enumerate(data_array.chunks[0]):
            dim0_start = idx * chunk_size
            dim0_end = (idx + 1) * chunk_size

            # reshape data array slice
            scr_data = da.zeros(
                (
                    dim0_end - dim0_start,
                    j_diff_max * num_tiles_y,
                    i_diff_max * num_tiles_x,
                ),
                chunks=(-1, j_diff_max, i_diff_max),
                dtype=data_array.dtype,
            )
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    scr_ij_bbox = scr_ij_bboxes[:, j, i]
                    scr_data[
                        :,
                        j * j_diff_max : (j + 1) * j_diff_max,
                        i * i_diff_max : (i + 1) * i_diff_max,
                    ] = data_array.data[
                        dim0_start:dim0_end,
                        scr_ij_bbox[1] : scr_ij_bbox[3],
                        scr_ij_bbox[0] : scr_ij_bbox[2],
                    ]

            data_reprojected = da.map_blocks(
                reproject_block,
                source_xx,
                source_yy,
                scr_data,
                x_coords,
                y_coords,
                dtype=data_array.dtype,
                chunks=(
                    dim0_end - dim0_start,
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


def reproject_block(
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
