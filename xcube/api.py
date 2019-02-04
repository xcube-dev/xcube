from contextlib import contextmanager
from typing import Dict

import xarray as xr

from xcube.dsio import find_dataset_io, guess_dataset_format, FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF4


@contextmanager
def open_dataset(input_path: str,
                 format_name: str = None,
                 **kwargs) -> xr.Dataset:
    """
    The ``read_dataset`` function as context manager that auto-closes the dataset read.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    dataset = read_dataset(input_path, format_name, **kwargs)
    try:
        yield dataset
    finally:
        dataset.close()


def read_dataset(input_path: str,
                 format_name: str = None,
                 **kwargs) -> xr.Dataset:
    """
    Read a dataset.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    format_name = format_name if format_name else guess_dataset_format(input_path)
    if format_name is None:
        raise ValueError("Unknown input format")
    dataset_io = find_dataset_io(format_name, modes=["r"])
    if dataset_io is None:
        raise ValueError(f"Unknown input format {format_name!r} for {input_path}")

    return dataset_io.read(input_path, **kwargs)


def write_dataset(dataset: xr.Dataset,
                  output_path: str,
                  format_name: str = None,
                  **kwargs) -> xr.Dataset:
    """
    Write dataset.

    :param dataset: Dataset to be written.
    :param output_path: output path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: the input dataset
    """
    format_name = format_name if format_name else guess_dataset_format(output_path)
    if format_name is None:
        raise ValueError("Unknown output format")
    dataset_io = find_dataset_io(format_name, modes=["w"])
    if dataset_io is None:
        raise ValueError(f"Unknown output format {format_name!r} for {output_path}")

    dataset_io.write(dataset, output_path, **kwargs)

    return dataset


def chunk_dataset(dataset: xr.Dataset,
                  chunk_sizes: Dict[str, int] = None,
                  format_name: str = None) -> xr.Dataset:
    """
    Chunk dataset and update encodings for given format.

    :param dataset: input dataset
    :param chunk_sizes: mapping from dimension name to new chunk size
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :return: the re-chunked dataset
    """
    chunked_ds = dataset.chunk(chunks=chunk_sizes)

    # Update encoding so writing of chunked_ds recognizes new chunks
    chunk_sizes_attr_name = None
    if format_name == FORMAT_NAME_ZARR:
        chunk_sizes_attr_name = "chunks"
    if format_name == FORMAT_NAME_NETCDF4:
        chunk_sizes_attr_name = "chunksizes"
    if chunk_sizes_attr_name:
        for var_name in chunked_ds.variables:
            var = chunked_ds[var_name]
            if chunk_sizes:
                sizes = tuple(chunk_sizes[dim_name] if dim_name in chunk_sizes
                              else var.shape[var.dims.index(dim_name)]
                              for dim_name in var.dims)
                var.encoding.update({chunk_sizes_attr_name: sizes})
            elif chunk_sizes_attr_name in var.encoding:
                # Remove any explicit and wrong specification so writing will use Dask chunks (TBC!)
                del var.encoding[chunk_sizes_attr_name]

    return chunked_ds


def dump_dataset(dataset: xr.Dataset,
                 variable_names=None,
                 show_var_encoding=False) -> str:
    """
    Dumps a dataset or variables into a text string.

    :param dataset: input dataset
    :param variable_names: names of variables to be dumped
    :param show_var_encoding: also dump variable encodings?
    :return: the dataset dump
    """
    lines = []
    if not variable_names:
        lines.append(str(dataset))
        if show_var_encoding:
            for var_name, var in dataset.coords.items():
                if var.encoding:
                    lines.append(dump_var_encoding(var, header=f"Encoding for coordinate variable {var_name!r}:"))
            for var_name, var in dataset.data_vars.items():
                if var.encoding:
                    lines.append(dump_var_encoding(var, header=f"Encoding for data variable {var_name!r}:"))
    else:
        for var_name in variable_names:
            var = dataset[var_name]
            lines.append(str(var))
            if show_var_encoding and var.encoding:
                lines.append(dump_var_encoding(var))
    return "\n".join(lines)


def dump_var_encoding(var: xr.DataArray, header="Encoding:", indent=4) -> str:
    """
    Dump the encoding information of a variable into a text string.

    :param var: Dataset variable.
    :param header: Title/header string.
    :param indent: Indention in spaces.
    :return: the variable dump
    """
    lines = [header]
    max_len = 0
    for k in var.encoding:
        max_len = max(max_len, len(k))
    indent_spaces = indent * " "
    for k, v in var.encoding.items():
        tab_spaces = (2 + max_len - len(k)) * " "
        lines.append(f"{indent_spaces}{k}:{tab_spaces}{v!r}")
    return "\n".join(lines)
