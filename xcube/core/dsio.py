# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import shutil
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import s3fs
import xarray as xr
import zarr

from xcube.constants import EXTENSION_POINT_DATASET_IOS
from xcube.constants import FORMAT_NAME_CSV, FORMAT_NAME_MEM, FORMAT_NAME_NETCDF4, FORMAT_NAME_ZARR
from xcube.core.timeslice import append_time_slice, insert_time_slice, replace_time_slice
from xcube.core.verify import assert_cube
from xcube.util.plugin import ExtensionComponent, get_extension_registry


def open_cube(input_path: str,
              format_name: str = None,
              **kwargs) -> xr.Dataset:
    """
    Open a xcube dataset from *input_path*.
    If *format* is not provided it will be guessed from *input_path*.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :return: xcube dataset
    """
    return open_dataset(input_path, format_name=format_name, is_cube=True, **kwargs)


def write_cube(cube: xr.Dataset,
               output_path: str,
               format_name: str = None,
               cube_asserted: bool = False,
               **kwargs) -> xr.Dataset:
    """
    Write a xcube dataset to *output_path*.
    If *format* is not provided it will be guessed from *output_path*.

    :param cube: xcube dataset to be written.
    :param output_path: output path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param kwargs: format-specific keyword arguments
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: xcube dataset *cube*
    """
    if not cube_asserted:
        assert_cube(cube)
    return write_dataset(cube, output_path, format_name=format_name, **kwargs)


def open_dataset(input_path: str,
                 format_name: str = None,
                 is_cube: bool = False,
                 **kwargs) -> xr.Dataset:
    """
    Open a dataset from *input_path*.
    If *format* is not provided it will be guessed from *output_path*.

    :param input_path: input path
    :param format_name: format, e.g. "zarr" or "netcdf4"
    :param is_cube: Whether a ValueError will be raised, if the dataset read from *input_path* is not a xcube dataset.
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


class DatasetIO(ExtensionComponent, metaclass=ABCMeta):
    """
    An abstract base class that represents dataset input/output.

    :param name: A unique dataset I/O identifier.
    """

    def __init__(self, name: str):
        super().__init__(EXTENSION_POINT_DATASET_IOS, name)

    @property
    def description(self) -> str:
        """
        :return: A description for this input processor
        """
        return self.get_metadata_attr('description', '')

    @property
    def ext(self) -> str:
        """The primary filename extension used by this dataset I/O."""
        return self.get_metadata_attr('ext', '')

    @property
    def modes(self) -> Set[str]:
        """
        A set describing the modes of this dataset I/O.
        Must be one or more of "r" (read), "w" (write), and "a" (append).
        """
        return self.get_metadata_attr('modes', set())

    @abstractmethod
    def fitness(self, path: str, path_type: str = None) -> float:
        """
        Compute a fitness of this dataset I/O in the interval [0 to 1]
        for reading/writing from/to the given *path*.

        :param path: The path or URL.
        :param path_type: Either "file", "dir", "url", or None.
        :return: the chance in range [0 to 1]
        """
        return 0.0

    def read(self, input_path: str, **kwargs) -> xr.Dataset:
        """Read a dataset from *input_path* using format-specific read parameters *kwargs*."""
        raise NotImplementedError()

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        """"Write *dataset* to *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        """"Append *dataset* to existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def insert(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        """"Insert *dataset* at *index* into existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def replace(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        """"Replace *dataset* at *index* in existing *output_path* using format-specific write parameters *kwargs*."""
        raise NotImplementedError()

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        """"Update *dataset* at *output_path* using format-specific open parameters *kwargs*."""
        raise NotImplementedError()


def get_extension(name: str):
    return get_extension_registry().get_extension(EXTENSION_POINT_DATASET_IOS, name)


def find_dataset_io_by_name(name: str):
    extension = get_extension(name)
    if not extension:
        return None
    return extension.component


def find_dataset_io(format_name: str, modes: Iterable[str] = None, default: DatasetIO = None) -> Optional[DatasetIO]:
    modes = set(modes) if modes else None
    format_name = format_name.lower()
    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)
    for dataset_io in dataset_ios:
        # noinspection PyUnresolvedReferences
        if format_name == dataset_io.name.lower():
            # noinspection PyTypeChecker
            if not modes or modes.issubset(dataset_io.modes):
                return dataset_io
    for dataset_io in dataset_ios:
        # noinspection PyUnresolvedReferences
        if format_name == dataset_io.ext.lower():
            # noinspection PyTypeChecker
            if not modes or modes.issubset(dataset_io.modes):
                return dataset_io
    return default


def guess_dataset_format(path: str) -> Optional[str]:
    """
    Guess a dataset format for a file system path or URL given by *path*.

    :param path: A file system path or URL.
    :return: The name of a dataset format guessed from *path*.
    """
    dataset_io_fitness_list = guess_dataset_ios(path)
    if dataset_io_fitness_list:
        return dataset_io_fitness_list[0][0].name
    return None


def guess_dataset_ios(path: str) -> List[Tuple[DatasetIO, float]]:
    """
    Guess suitable DatasetIO objects for a file system path or URL given by *path*.

    Returns a list of (DatasetIO, fitness) tuples, sorted by descending fitness values.
    Fitness values are in the interval (0, 1].

    The first entry is the most appropriate DatasetIO object.

    :param path: A file system path or URL.
    :return: A list of (DatasetIO, fitness) tuples.
    """
    if os.path.isfile(path):
        input_type = "file"
    elif os.path.isdir(path):
        input_type = "dir"
    elif path.find("://") > 0:
        input_type = "url"
    else:
        input_type = None

    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)

    dataset_io_fitness_list = []
    for dataset_io in dataset_ios:
        fitness = dataset_io.fitness(path, path_type=input_type)
        if fitness > 0.0:
            dataset_io_fitness_list.append((dataset_io, fitness))

    dataset_io_fitness_list.sort(key=lambda item: -item[1])
    return dataset_io_fitness_list


def _get_ext(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    return ext.lower()


def query_dataset_io(filter_fn: Callable[[DatasetIO], bool] = None) -> List[DatasetIO]:
    dataset_ios = get_extension_registry().find_components(EXTENSION_POINT_DATASET_IOS)
    if filter_fn is None:
        return dataset_ios
    return list(filter(filter_fn, dataset_ios))


# noinspection PyAbstractClass
class MemDatasetIO(DatasetIO):
    """
    An in-memory  dataset I/O. Keeps all datasets in a dictionary.

    :param datasets: The initial datasets as a path to dataset mapping.
    """

    def __init__(self, datasets: Dict[str, xr.Dataset] = None):
        super().__init__(FORMAT_NAME_MEM)
        self._datasets = datasets or {}

    @property
    def datasets(self) -> Dict[str, xr.Dataset]:
        return self._datasets

    def fitness(self, path: str, path_type: str = None) -> float:
        if path in self._datasets:
            return 1.0
        ext_value = _get_ext(path) == ".mem"
        type_value = 0.0
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        if path in self._datasets:
            return self._datasets[path]
        raise FileNotFoundError(path)

    def write(self, dataset: xr.Dataset, path: str, **kwargs):
        self._datasets[path] = dataset

    def append(self, dataset: xr.Dataset, path: str, **kwargs):
        if path in self._datasets:
            old_ds = self._datasets[path]
            # noinspection PyTypeChecker
            self._datasets[path] = xr.concat([old_ds, dataset],
                                             dim='time',
                                             data_vars='minimal',
                                             coords='minimal',
                                             compat='equals')
        else:
            self._datasets[path] = dataset.copy()

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            ds = self._datasets[output_path]
            ds.attrs.update(global_attrs)


class Netcdf4DatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to NetCDF files.
    """

    def __init__(self):
        super().__init__(FORMAT_NAME_NETCDF4)

    def fitness(self, path: str, path_type: str = None) -> float:
        ext = _get_ext(path)
        ext_value = ext in {'.nc', '.hdf', '.h5'}
        type_value = 0.0
        if path_type is "file":
            type_value = 1.0
        elif path_type is None:
            type_value = 0.5
        else:
            ext_value = 0.0
        return (3 * ext_value + type_value) / 4

    def read(self, input_path: str, **kwargs) -> xr.Dataset:
        return xr.open_dataset(input_path, **kwargs)

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        dataset.to_netcdf(output_path)

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        import os
        temp_path = output_path + '.temp.nc'
        os.rename(output_path, temp_path)
        old_ds = xr.open_dataset(temp_path)
        new_ds = xr.concat([old_ds, dataset],
                           dim='time',
                           data_vars='minimal',
                           coords='minimal',
                           compat='equals')
        # noinspection PyUnresolvedReferences
        new_ds.to_netcdf(output_path)
        old_ds.close()
        rimraf(temp_path)

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            import netCDF4
            ds = netCDF4.Dataset(output_path, 'r+')
            ds.setncatts(global_attrs)
            ds.close()


class ZarrDatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to Zarr directories or archives.
    """

    def __init__(self):
        super().__init__(FORMAT_NAME_ZARR)

    def fitness(self, path: str, path_type: str = None) -> float:
        ext = _get_ext(path)
        ext_value = 0.0
        type_value = 0.0
        if ext == ".zarr":
            ext_value = 1.0
            if path_type is "dir":
                type_value = 1.0
            elif path_type == "url" or path_type is None:
                type_value = 0.5
            else:
                ext_value = 0.0
        else:
            lower_path = path.lower()
            if lower_path.endswith(".zarr.zip"):
                ext_value = 1.0
                if path_type == "file":
                    type_value = 1.0
                elif path_type is None:
                    type_value = 0.5
                else:
                    ext_value = 0.0
            else:
                if path_type is "dir":
                    type_value = 1.0
                elif path_type == "url":
                    type_value = 0.5
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        path_or_store = path
        consolidated = False
        mode = "read"
        root = None

        if isinstance(path, str):
            client_kwargs = {}
            if "client_kwargs" in kwargs:
                client_kwargs = kwargs.pop('client_kwargs')
            if 'endpoint_url' in kwargs:
                client_kwargs['endpoint_url'] = kwargs.pop('endpoint_url')
                root = path
            if 'region_name' in kwargs:
                client_kwargs['region_name'] = kwargs.pop('region_name')

            path_or_store, root = _get_path_or_store(path_or_store, client_kwargs, mode, root)

            if "endpoint_url" in client_kwargs and root is not None:
                s3 = s3fs.S3FileSystem(anon=True,
                                       client_kwargs=client_kwargs)
                consolidated = s3.exists(f'{root}/.zmetadata')
                path_or_store = s3fs.S3Map(root=root, s3=s3, check=False)
                if 'max_cache_size' in kwargs:
                    max_cache_size = kwargs.pop('max_cache_size')
                    if max_cache_size > 0:
                        path_or_store = zarr.LRUStoreCache(path_or_store, max_size=max_cache_size)
            else:
                consolidated = os.path.exists(os.path.join(path_or_store, '.zmetadata'))
        return xr.open_zarr(path_or_store, consolidated=consolidated, **kwargs)

    def write(self,
              dataset: xr.Dataset,
              output_path: str,
              compress=True,
              cname=None, clevel=None, shuffle=None, blocksize=None,
              chunksizes=None,
              client_kwargs=None,
              **kwargs):
        mode = "write"
        root = None
        path_or_store, root = _get_path_or_store(output_path, client_kwargs, mode, root)

        # keeping the code commented out below, because might be useful for makinging write able to change bucket policy
        # if 'acl' in kwargs:
        #     acl = kwargs.pop('acl')
        encoding = self._get_write_encodings(dataset, compress, cname, clevel, shuffle, blocksize, chunksizes)
        dataset.to_zarr(path_or_store, mode="w", encoding=encoding)
        # if acl and client_kwargs and s3:
        #     s3.chmod(path=root, acl=acl, kwargs=client_kwargs)

    @classmethod
    def _get_write_encodings(cls, dataset, compress, cname, clevel, shuffle, blocksize, chunksizes):
        encoding = None
        if chunksizes:
            encoding = {}
            for var_name in dataset.data_vars:
                var = dataset[var_name]
                chunks: List[int] = []
                for i in range(len(var.dims)):
                    dim_name = var.dims[i]
                    if dim_name in chunksizes:
                        chunks.append(chunksizes[dim_name])
                    else:
                        chunks.append(var.shape[i])
                encoding[var_name] = dict(chunks=chunks)
        if compress:
            blosc_kwargs = dict(cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize)
            for k in list(blosc_kwargs.keys()):
                if blosc_kwargs[k] is None:
                    del blosc_kwargs[k]
            compressor = zarr.Blosc(**blosc_kwargs)

            if encoding:
                for var_name in encoding.keys():
                    encoding[var_name].update(compressor=compressor)
            else:
                encoding = {var_name: dict(compressor=compressor) for var_name in dataset.data_vars}
        return encoding

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        append_time_slice(output_path, dataset)

    def insert(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        insert_time_slice(output_path, index, dataset)

    def replace(self, dataset: xr.Dataset, index: int, output_path: str, **kwargs):
        replace_time_slice(output_path, index, dataset)

    def update(self, output_path: str, global_attrs: Dict[str, Any] = None, **kwargs):
        if global_attrs:
            import zarr
            ds = zarr.open_group(output_path, mode='r+', **kwargs)
            ds.attrs.update(global_attrs)


# noinspection PyAbstractClass
class CsvDatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to CSV files.
    """

    def __init__(self):
        super().__init__(FORMAT_NAME_CSV)

    def fitness(self, path: str, path_type: str = None) -> float:
        if path_type == "dir":
            return 0.0
        ext = _get_ext(path)
        ext_value = {".csv": 1.0, ".txt": 0.5, ".dat": 0.2}.get(ext, 0.1)
        type_value = {"file": 1.0, "url": 0.5, None: 0.5}.get(path_type, 0.0)
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        return xr.Dataset.from_dataframe(pd.read_csv(path, **kwargs))

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        dataset.to_dataframe().to_csv(output_path, **kwargs)


def rimraf(path):
    """
    The UNIX command `rm -rf` for xcube.
    Recursively remove directory or single file.

    :param path:  directory or single file
    """
    if os.path.isdir(path):
        try:
            shutil.rmtree(path, ignore_errors=False)
        except OSError:
            warnings.warn(f"failed to remove file {path}")
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            warnings.warn(f"failed to remove file {path}")
            pass


def _get_path_or_store(path, client_kwargs, mode, root):
    path_or_store = path
    anon_mode = True

    aws_credentials = ['aws_access_key_id', 'aws_secret_access_key']
    for credential_i in aws_credentials:
        try:
            credential_i_value = os.getenv(credential_i)
            if credential_i not in client_kwargs:
                client_kwargs[credential_i] = credential_i_value
        except KeyError:
            pass

    if client_kwargs is not None:
        if 'aws_access_key_id' in client_kwargs and 'aws_secret_access_key' in client_kwargs:
            anon_mode = False
    if path.startswith("https://") or path.startswith("http://"):
        import urllib3.util
        url = urllib3.util.parse_url(path_or_store)
        if url.port is not None:
            client_kwargs['endpoint_url'] = f'{url.scheme}://{url.host}:{url.port}'
        else:
            client_kwargs['endpoint_url'] = f'{url.scheme}://{url.host}'
        root = url.path
        if root.startswith('/'):
            root = root[1:]
        if "write" in mode:
            root = f's3://{root}'
        s3 = s3fs.S3FileSystem(anon=anon_mode,
                               client_kwargs=client_kwargs)
        path_or_store = s3fs.S3Map(root=root, s3=s3, check=False)
    return path_or_store, root
