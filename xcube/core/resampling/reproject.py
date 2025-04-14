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
            da_slice = data_array[dim0_start:dim0_end, ...]

            def reproject_block(
                source_xx: np.ndarray,
                source_yy: np.ndarray,
                fill_value: int | float,
                interpolation: int,
                block_id=None,
            ):
                scr_ij_bbox = scr_ij_bboxes[:, block_id[1], block_id[2]]
                if np.all(scr_ij_bbox == -1):
                    return np.full(
                        (da_slice.shape[0], *source_xx.shape),
                        fill_value,
                        dtype=data_array.dtype,
                    )
                da_clip = da_slice[
                    :, scr_ij_bbox[1] : scr_ij_bbox[3], scr_ij_bbox[0] : scr_ij_bbox[2]
                ]
                x_name, y_name = source_gm.xy_dim_names
                y = da_clip[y_name].values
                x = da_clip[x_name].values
                data = da_clip.values

                ix = (source_xx - x[0]) / source_gm.x_res
                iy = (source_yy - y[0]) / -source_gm.y_res

                if interpolation == 0:
                    ix = np.rint(ix).astype(np.int16)
                    iy = np.rint(iy).astype(np.int16)
                    data_reprojected = data[:, iy, ix]
                else:
                    raise NotImplementedError()

                return data_reprojected

            data_reprojected = da.map_blocks(
                reproject_block,
                source_xx,
                source_yy,
                dtype=data_array.dtype,
                chunks=(
                    da_slice.chunks[0][0],
                    source_yy.chunks[0][0],
                    source_yy.chunks[1][0],
                ),
                fill_value=fill_value,
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
