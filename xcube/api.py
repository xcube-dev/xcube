import os
from typing import Dict, Optional

import xarray as xr

FORMAT_NAME_ZARR = "zarr"
FORMAT_NAME_NETCDF = "netcdf"

from contextlib import contextmanager


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
    ds = read_dataset(input_path, format_name, **kwargs)
    try:
        yield ds
    finally:
        ds.close()


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
    format_name = format_name if format_name is not None else guess_dataset_format(input_path)
    # TODO (forman): allow opening from URLs too
    if format_name == FORMAT_NAME_ZARR:
        ds = xr.open_zarr(input_path, **kwargs)
    else:
        ds = xr.open_dataset(input_path, **kwargs)
    return ds


def write_dataset(ds: xr.Dataset,
                  output_path: str,
                  format_name: str = None,
                  **kwargs) -> xr.Dataset:
    """
    Write dataset.

    :param ds: input dataset
    :param output_path: output path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: the input dataset
    """
    _validate_dataset_format(format_name)
    format_name = format_name if format_name is not None else guess_dataset_format(output_path)
    # TODO (forman): allow writing to URLs too
    if format_name == FORMAT_NAME_ZARR:
        if "mode" not in kwargs:
            kwargs["mode"] = "w"
        ds = ds.to_zarr(output_path, **kwargs)
    else:
        ds = ds.to_netcdf(output_path, **kwargs)
    return ds


def chunk_dataset(ds: xr.Dataset,
                  chunk_sizes: Dict[str, int] = None,
                  format_name: str = None) -> xr.Dataset:
    """
    Chunk dataset and update encodings for given format.

    :param ds: input dataset
    :param chunk_sizes: mapping from dimension name to new chunk size
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :return: the re-chunked dataset
    """
    _validate_dataset_format(format_name)

    chunked_ds = ds.chunk(chunks=chunk_sizes)

    # Update encoding so writing of chunked_ds recognizes new chunks
    chunk_sizes_attr_name = None
    if format_name == FORMAT_NAME_ZARR:
        chunk_sizes_attr_name = "chunks"
    if format_name == FORMAT_NAME_NETCDF:
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


def dump_dataset(ds: xr.Dataset,
                 variable_names=None,
                 show_var_encoding=False) -> xr.Dataset:
    """
    Dumps a dataset or variables contained in it to stdout.

    :param ds: input dataset
    :param variable_names: names of variables to be dumped
    :param show_var_encoding: also dump variable encodings?
    :return: the input dataset
    """
    if not variable_names:
        print(ds)
        if show_var_encoding:
            for var_name, var in ds.coords.items():
                if var.encoding:
                    dump_var_encoding(var, header=f"Encoding for coordinate variable {var_name!r}:")
            for var_name, var in ds.data_vars.items():
                if var.encoding:
                    dump_var_encoding(var, header=f"Encoding for data variable {var_name!r}:")
    else:
        for var_name in variable_names:
            var = ds[var_name]
            print(var)
            if show_var_encoding and var.encoding:
                dump_var_encoding(var)

    return ds


def dump_var_encoding(var: xr.DataArray, header="Encoding:", indent=4):
    """
    Dump the encoding information of a variable to stdout.

    :param var: Dataset variable.
    :param header: Title/header string.
    :param indent: Indention in spaces.
    """
    print(header)
    max_len = 0
    for k in var.encoding:
        max_len = max(max_len, len(k))
    indent_spaces = indent * " "
    for k, v in var.encoding.items():
        tab_spaces = (2 + max_len - len(k)) * " "
        print(f"{indent_spaces}{k}:{tab_spaces}{v!r}")


def guess_dataset_format(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in {'.zarr', '.zarr.zip'}:
        return FORMAT_NAME_ZARR
    if ext in {'.nc', '.hdf', '.h5'}:
        return FORMAT_NAME_NETCDF
    return None


def _validate_dataset_format(format_name: str = None):
    if format_name not in {None, FORMAT_NAME_ZARR, FORMAT_NAME_NETCDF}:
        raise ValueError('Invalid format: {format!r}')
