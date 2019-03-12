import os
from typing import List, Callable, Sequence, Optional

import xarray as xr

from .readwrite import read_dataset

PyramidLevelCallback = Callable[[xr.Dataset, int, int], Optional[xr.Dataset]]


def compute_pyramid_levels(dataset: xr.Dataset,
                           dims: Sequence[str] = None,
                           shape: Sequence[int] = None,
                           chunks: Sequence[int] = None,
                           var_names: Sequence[str] = None,
                           max_num_levels: int = None,
                           post_process_level: PyramidLevelCallback = None,
                           progress_monitor: PyramidLevelCallback = None) -> List[xr.Dataset]:
    if var_names:
        dropped_vars = list(set(dataset.data_vars).difference(set(var_names)))
    else:
        dropped_vars = []

    for var_name in dataset.data_vars:
        var = dataset[var_name]

        var_shape = var.shape
        if len(var_shape) < 2:
            dropped_vars.append(var_name)
            continue

        if shape is None:
            shape = var_shape
        elif shape != var_shape:
            dropped_vars.append(var_name)
            continue

        var_dims = var.dims
        if dims is None:
            dims = var_dims
        elif dims != var_dims:
            dropped_vars.append(var_name)
            continue

        var_chunks = var.chunks
        if chunks is None:
            chunks = var_chunks

    if dropped_vars:
        dataset = dataset.drop(dropped_vars)

    var_names = list(dataset.data_vars)
    if not var_names or shape is None:
        raise ValueError("Cannot pyramidize because no suitable variables were found.")

    if chunks is None:
        raise ValueError("Cannot pyramidize because chunk sizes were not given and none could be found.")

    if shape and chunks and len(shape) != len(chunks):
        raise ValueError("Cannot pyramidize because of inconsistent chunking.")

    def chunk_to_int(chunk):
        return chunk if isinstance(chunk, int) else max(chunk)

    chunks = tuple(map(chunk_to_int, chunks))

    ch, cw = chunks[-2:]

    # Count num_levels
    h, w = shape[-2:]
    num_levels = 0
    while True:
        num_levels += 1

        w = (w + 1) // 2
        h = (h + 1) // 2
        if w < cw or h < ch \
                or (max_num_levels is not None and max_num_levels == num_levels):
            break

    # Compute levels
    h, w = shape[-2:]
    level_dataset = dataset
    level_datasets = []
    while True:
        if post_process_level is not None:
            level_dataset = post_process_level(level_dataset, len(level_datasets), num_levels)
        if progress_monitor is not None:
            progress_monitor(level_dataset, len(level_datasets), num_levels)
        level_datasets.append(level_dataset)

        w = (w + 1) // 2
        h = (h + 1) // 2
        if w < cw or h < ch \
                or (max_num_levels is not None and max_num_levels == len(level_datasets)):
            break

        downsampled_vars = {}
        for var_name in level_dataset.data_vars:
            var = level_dataset.data_vars[var_name]
            # For time being, we use the simplest and likely fastest downsampling I can think of
            downsampled_var = var[..., ::2, ::2]
            downsampled_var = downsampled_var.chunk(chunks=chunks)
            downsampled_var.encoding.update(chunks=chunks)
            downsampled_vars[var_name] = downsampled_var

        level_dataset = xr.Dataset(downsampled_vars)

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
