from contextlib import contextmanager

import xarray as xr

from .verify import assert_cube
from xcube.util.dsio import find_dataset_io, guess_dataset_format


@contextmanager
def open_cube(input_path: str,
              format_name: str = None,
              **kwargs) -> xr.Dataset:
    """
    The ``read_cube`` function as context manager that auto-closes the cube read.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: data cube
    """
    dataset = read_cube(input_path, format_name, **kwargs)
    try:
        yield dataset
    finally:
        dataset.close()


def read_cube(input_path: str,
              format_name: str = None,
              **kwargs) -> xr.Dataset:
    """
    Read a data cube from *input_path*.
    If *format* is not provided it will be guessed from *input_path*.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: data cube
    """
    return read_dataset(input_path, format_name=format_name, is_cube=True, **kwargs)


def write_cube(cube: xr.Dataset,
               output_path: str,
               format_name: str = None,
               cube_asserted: bool = False,
               **kwargs) -> xr.Dataset:
    """
    Write a data cube to *output_path*.
    If *format* is not provided it will be guessed from *output_path*.

    :param cube: Data cube to be written.
    :param output_path: output path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: data cube *cube*
    """
    if not cube_asserted:
        assert_cube(cube)
    return write_dataset(cube, output_path, format_name=format_name, **kwargs)


@contextmanager
def open_dataset(input_path: str,
                 format_name: str = None,
                 is_cube: bool = False,
                 **kwargs) -> xr.Dataset:
    """
    The ``read_dataset`` function as context manager that auto-closes the dataset read.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param is_cube: Whether a ValueError will be raised, if the dataset read from *input_path* is not a data cube.
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    dataset = read_dataset(input_path, format_name, is_cube=is_cube, **kwargs)
    try:
        yield dataset
    finally:
        dataset.close()


def read_dataset(input_path: str,
                 format_name: str = None,
                 is_cube: bool = False,
                 **kwargs) -> xr.Dataset:
    """
    Read dataset from *input_path*.
    If *format* is not provided it will be guessed from *output_path*.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param is_cube: Whether a ValueError will be raised, if the dataset read from *input_path* is not a data cube.
    :param kwargs: format-specific keyword arguments
    :return: dataset object
    """
    format_name = format_name if format_name else guess_dataset_format(input_path)
    if format_name is None:
        raise ValueError("Unknown input format")
    dataset_io = find_dataset_io(format_name, modes=["r"])
    if dataset_io is None:
        raise ValueError(f"Unknown input format {format_name!r} for {input_path}")
    dataset = dataset_io.read(input_path, **kwargs)
    if is_cube:
        assert_cube(dataset)
    return dataset


def write_dataset(dataset: xr.Dataset,
                  output_path: str,
                  format_name: str = None,
                  **kwargs) -> xr.Dataset:
    """
    Write dataset to *output_path*.
    If *format* is not provided it will be guessed from *output_path*.

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
