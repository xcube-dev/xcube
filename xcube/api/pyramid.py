import os
from typing import List, Callable, Sequence, Optional, Tuple

import xarray as xr

from .readwrite import read_dataset

PyramidLevelCallback = Callable[[xr.Dataset, int, int], Optional[xr.Dataset]]


def compute_pyramid_levels(dataset: xr.Dataset,
                           spatial_dims: Tuple[str, str] = None,
                           spatial_shape: Tuple[int, int] = None,
                           spatial_tile_shape: Tuple[int, int] = None,
                           var_names: Sequence[str] = None,
                           max_num_levels: int = None,
                           post_process_level: PyramidLevelCallback = None,
                           progress_monitor: PyramidLevelCallback = None) -> List[xr.Dataset]:
    """
    Compute the pyramid level datasets for a given *dataset*.

    It is assumed that the datasets data variables

    :param dataset:
    :param spatial_dims:
    :param spatial_shape:
    :param spatial_tile_shape:
    :param var_names:
    :param max_num_levels:
    :param post_process_level:
    :param progress_monitor:
    :return:
    """
    if var_names:
        var_names = set(var_names)
        dropped_vars = list(set(dataset.data_vars).difference(var_names))
    else:
        var_names = set(dataset.data_vars)
        dropped_vars = []

    # Collect data variables to be dropped, derive missing information from spatial data variables
    for var_name in var_names:

        if var_name not in dataset.data_vars:
            raise ValueError(f"variable {var_name} not found")

        var = dataset[var_name]

        if var.ndim < 2:
            # Must have at least the two spatial dimensions
            dropped_vars.append(var_name)
            continue

        if spatial_dims is None:
            spatial_dims = var.dims[-2:]
        elif spatial_dims != var.dims[-2:]:
            # Spatial dimensions don't fit
            dropped_vars.append(var_name)
            continue

        if spatial_shape is None:
            spatial_shape = var.shape[-2:]
        elif spatial_shape != var.shape[-2:]:
            # Spatial dimension sizes don't fit
            dropped_vars.append(var_name)
            continue

        if spatial_tile_shape is None and var.chunks is not None:
            def chunk_to_int(chunk):
                return chunk if isinstance(chunk, int) else max(chunk)

            spatial_tile_shape = tuple(map(chunk_to_int, var.chunks[-2:]))

    if dropped_vars:
        dataset = dataset.drop(dropped_vars)

    if not tuple(dataset.data_vars):
        raise ValueError("Cannot pyramidize because no suitable variables were found.")

    if spatial_tile_shape is None:
        spatial_tile_shape = min(spatial_shape[0], 512), min(spatial_shape[1], 512)

    tile_height, tile_width = spatial_tile_shape

    # Count num_levels
    height, width = spatial_shape
    num_levels = 0
    while True:
        num_levels += 1

        width = (width + 1) // 2
        height = (height + 1) // 2
        if width < tile_width or height < tile_height \
                or (max_num_levels is not None and max_num_levels == num_levels):
            break

    # Compute levels
    height, width = spatial_shape
    level_dataset = dataset
    level_datasets = []
    for level in range(num_levels):
        if level > 0:
            # Down-sample levels
            downsampled_vars = {}
            for var_name in level_dataset.data_vars:
                var = level_dataset.data_vars[var_name]
                # For time being, we use the simplest and likely fastest downsampling I can think of
                downsampled_vars[var_name] = var[..., ::2, ::2]
            level_dataset = xr.Dataset(downsampled_vars, attrs=level_dataset.attrs)

        # Chunk variables in level dataset according to spatial_tile_shape
        chunked_vars = {}
        for var_name in level_dataset.data_vars:
            var = level_dataset.data_vars[var_name]
            height, width = var.shape[-2:]
            zarr_chunks = (1,) * (var.ndim - 2) + (tile_height,) + (tile_width,)
            dask_chunks = (1,) * (var.ndim - 2) + (_tile_chunk(height, tile_height),) + (_tile_chunk(width, tile_width),)
            dask_chunks = {var.dims[i]: dask_chunks[i] for i in range(var.ndim)}
            chunked_var = var.chunk(chunks=dask_chunks)
            chunked_var.encoding.update(chunks=zarr_chunks)
            chunked_vars[var_name] = chunked_var
        level_dataset = level_dataset.assign(chunked_vars)

        # Apply post processor, if any
        if post_process_level is not None:
            level_dataset = post_process_level(level_dataset, len(level_datasets), num_levels)

        # Inform progress monitor, if any
        if progress_monitor is not None:
            progress_monitor(level_dataset, len(level_datasets), num_levels)

        # Collect level dataset
        level_datasets.append(level_dataset)

        # Compute new size
        width = (width + 1) // 2
        height = (height + 1) // 2

    return level_datasets


def write_pyramid_levels(output_path: str,
                         dataset: xr.Dataset = None,
                         input_path: str = None,
                         link_input: bool = False,
                         progress_monitor: PyramidLevelCallback = None,
                         **kwargs):
    if dataset is None and input_path is None:
        raise ValueError("at least one of dataset or input_path must be given")

    if link_input and input_path is None:
        raise ValueError("input_path must be provided to link input")

    _post_process_level = kwargs.pop("post_process_level") if "post_process_level" in kwargs else None

    def post_process_level(level_dataset, index, num_levels):
        if _post_process_level is not None:
            level_dataset = _post_process_level(level_dataset, index, num_levels)

        if index == 0 and link_input:
            with open(os.path.join(output_path, f"{index}.link"), "w") as fp:
                fp.write(input_path)
        else:
            path = os.path.join(output_path, f"{index}.zarr")
            level_dataset.to_zarr(path)
            level_dataset.close()
            level_dataset = xr.open_zarr(path)

        if progress_monitor is not None:
            progress_monitor(level_dataset, index, num_levels)

        return level_dataset

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if dataset is None:
        dataset = read_dataset(input_path)

    return compute_pyramid_levels(dataset, post_process_level=post_process_level, **kwargs)


def read_pyramid_levels(dir_path: str,
                        progress_monitor: PyramidLevelCallback = None) -> List[xr.Dataset]:
    file_paths = os.listdir(dir_path)
    level_paths = {}
    num_levels = -1
    for filename in file_paths:
        file_path = os.path.join(dir_path, filename)
        basename, ext = os.path.splitext(filename)
        if basename.isdigit():
            index = int(basename)
            num_levels = max(num_levels, index + 1)
            if os.path.isfile(file_path) and ext == ".link":
                level_paths[index] = (ext, file_path)
            elif os.path.isdir(file_path) and ext == ".zarr":
                level_paths[index] = (ext, file_path)

    if num_levels != len(level_paths):
        raise ValueError(f"Inconsistent pyramid directory:"
                         f" expected {num_levels} but found {len(level_paths)} entries:"
                         f" {dir_path}")

    levels = []
    for index in range(num_levels):
        ext, file_path = level_paths[index]
        if ext == ".link":
            with open(file_path, "r") as fp:
                link_file_path = fp.read()
            dataset = xr.open_zarr(link_file_path)
        else:
            dataset = xr.open_zarr(file_path)
        if progress_monitor is not None:
            progress_monitor(dataset, index, num_levels)
        levels.append(dataset)
    return levels


def _tile_chunk(size, tile_size):
    last_tile_size = size % tile_size
    if last_tile_size != 0:
        return (tile_size,) * (size // tile_size) + (last_tile_size,)
    return tile_size
