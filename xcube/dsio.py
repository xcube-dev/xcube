# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

import glob
import os
from abc import abstractmethod, ABCMeta
from typing import Set, Callable, List, Optional, Dict, Iterable, Tuple

import pandas as pd
import s3fs
import xarray as xr
import zarr

from xcube.objreg import get_obj_registry

FORMAT_NAME_EXCEL = "excel"
FORMAT_NAME_CSV = "csv"
FORMAT_NAME_MEM = "mem"
FORMAT_NAME_NETCDF4 = "netcdf4"
FORMAT_NAME_ZARR = "zarr"


def open_from_fs(paths: str, recursive: bool = False, **kwargs):
    """
    Open an xcube (xarray dataset) from local or any mounted file system.

    :param paths: A path which may contain Unix-style wildcards or a sequence of paths.
    :param recursive: Whether wildcards should be resolved recursively in sub-directories.
    :type kwargs: keyword arguments passed to `xarray.open_mfdataset()`.
    """
    if isinstance(paths, str):
        paths = [file for file in glob.glob(paths, recursive=recursive)]
        if 'autoclose' not in kwargs:
            kwargs['autoclose'] = True

    if 'coords' not in kwargs:
        kwargs['coords'] = 'minimum'
    if 'data_vars' not in kwargs:
        kwargs['data_vars'] = 'minimum'

    ds = xr.open_mfdataset(paths, **kwargs)

    if 'chunks' not in kwargs:
        print("ds.encoding = ", ds.encoding)

    return ds


def open_from_obs(path: str, endpoint_url: str = None, max_cache_size: int = 2 ** 28) -> xr.Dataset:
    """
    Open an xcube (xarray dataset) from S3 compatible object storage (OBS).

    :param path: Path having format "<bucket>/<my>/<sub>/<path>"
    :param endpoint_url: Optional URL of the OBS service endpoint. If omitted, AWS S3 service URL is used.
    :param max_cache_size: If > 0, size of a memory cache in bytes, e.g. 2**30 = one giga bytes.
           If None or size <= 0, no memory cache will be used.
    :return: an xarray dataset
    """
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=endpoint_url))
    store = s3fs.S3Map(root=path, s3=s3, check=False)
    if max_cache_size is not None and max_cache_size > 0:
        store = zarr.LRUStoreCache(store, max_size=max_cache_size)
    return xr.open_zarr(store)


class DatasetIO(metaclass=ABCMeta):
    """
    An abstract base class that represents dataset input/output.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this dataset I/O."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description for this dataset I/O."""
        pass

    @property
    @abstractmethod
    def ext(self) -> str:
        """The primary filename extension used by this dataset I/O."""
        pass

    @property
    @abstractmethod
    def modes(self) -> Set[str]:
        """
        A set describing the modes of this dataset I/O.
        Must be one or more of "r" (read), "w" (write), and "a" (append).
        """
        pass

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


def register_dataset_io(dataset_io: DatasetIO):
    # noinspection PyTypeChecker
    get_obj_registry().put(dataset_io.name, dataset_io, type=DatasetIO)


def find_dataset_io(format_name: str, modes: Iterable[str] = None, default: DatasetIO = None) -> Optional[DatasetIO]:
    modes = set(modes) if modes else None
    format_name = format_name.lower()
    dataset_ios = get_obj_registry().get_all(type=DatasetIO)
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

    dataset_ios = get_obj_registry().get_all(type=DatasetIO)

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
    dataset_ios = get_obj_registry().get_all(type=DatasetIO)
    if filter_fn is None:
        return dataset_ios
    return list(filter(filter_fn, dataset_ios))


class MemDatasetIO(DatasetIO):
    """
    An in-memory  dataset I/O. Keeps all datasets in a dictionary.

    :param datasets: The initial datasets as a path to dataset mapping.
    """

    def __init__(self, datasets: Dict[str, xr.Dataset] = None):
        self.datasets = datasets or {}

    @property
    def name(self) -> str:
        return FORMAT_NAME_MEM

    @property
    def description(self) -> str:
        return 'In-memory dataset I/O'

    @property
    def ext(self) -> str:
        return 'mem'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w', 'a'}

    def fitness(self, path: str, path_type: str = None) -> float:
        if path in self.datasets:
            return 1.0
        ext_value = _get_ext(path) == ".mem"
        type_value = 0.0
        return (3 * ext_value + type_value) / 4

    def read(self, path: str, **kwargs) -> xr.Dataset:
        if path in self.datasets:
            return self.datasets[path]
        raise FileNotFoundError(path)

    def write(self, dataset: xr.Dataset, path: str, **kwargs):
        self.datasets[path] = dataset

    def append(self, dataset: xr.Dataset, path: str, **kwargs):
        if path in self.datasets:
            old_ds = self.datasets[path]
            # noinspection PyTypeChecker
            self.datasets[path] = xr.concat([old_ds, dataset],
                                            dim='time',
                                            data_vars='minimal',
                                            coords='minimal',
                                            compat='equals')
        else:
            self.datasets[path] = dataset.copy()


class Netcdf4DatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to NetCDF files.
    """

    @property
    def name(self) -> str:
        return FORMAT_NAME_NETCDF4

    @property
    def description(self) -> str:
        return 'NetCDF-4 file format'

    @property
    def ext(self) -> str:
        return 'nc'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w', 'a'}

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
        temp_path = output_path + 'temp.nc'
        os.rename(output_path, temp_path)
        old_ds = xr.open_dataset(temp_path, decode_times=False)
        new_ds = xr.concat([old_ds, dataset],
                           dim='time',
                           data_vars='minimal',
                           coords='minimal',
                           compat='equals')
        # noinspection PyUnresolvedReferences
        new_ds.to_netcdf(output_path)
        old_ds.close()
        rimraf(temp_path)


class ZarrDatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to Zarr directories or archives.
    """

    def __init__(self):
        self.root_group = None

    @property
    def name(self) -> str:
        return FORMAT_NAME_ZARR

    @property
    def description(self) -> str:
        return 'Zarr file format (http://zarr.readthedocs.io)'

    @property
    def ext(self) -> str:
        return 'zarr'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w', 'a'}

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
        return xr.open_zarr(path, **kwargs)

    def write(self, dataset: xr.Dataset, output_path: str,
              compress=True,
              cname=None, clevel=None, shuffle=None, blocksize=None,
              chunksizes=None):

        encoding = {}
        if compress:
            blosc_kwargs = dict(cname=cname, clevel=clevel, shuffle=shuffle, blocksize=blocksize)
            for k in list(blosc_kwargs.keys()):
                if blosc_kwargs[k] is None:
                    del blosc_kwargs[k]
            encoding["compressor"] = zarr.Blosc(**blosc_kwargs)

        if chunksizes:
            chunks = []
            for dim_name, dim_size in dataset.dims.items():
                if dim_name in chunksizes:
                    chunks.append(chunksizes[dim_name])
                else:
                    chunks.append(dim_size)
            encoding["chunks"] = tuple(chunks)

        # Apply encodings to all variables
        var_encodings = {var_name: encoding for var_name in dataset.data_vars}

        dataset.to_zarr(output_path, mode="w", encoding=var_encodings)

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        import zarr
        if self.root_group is None:
            self.root_group = zarr.open(output_path, mode='a')
        for var_name, var_array in self.root_group.arrays():
            new_var = dataset[var_name]
            if 'time' in new_var.dims:
                axis = new_var.dims.index('time')
                var_array.append(new_var, axis=axis)


class CsvDatasetIO(DatasetIO):
    """
    A dataset I/O that reads from / writes to CSV files.
    """

    def __init__(self):
        self.root_group = None

    @property
    def name(self) -> str:
        return FORMAT_NAME_CSV

    @property
    def description(self) -> str:
        return 'CSV file format'

    @property
    def ext(self) -> str:
        return 'csv'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w'}

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


register_dataset_io(CsvDatasetIO())
register_dataset_io(MemDatasetIO())
register_dataset_io(Netcdf4DatasetIO())
register_dataset_io(ZarrDatasetIO())


def rimraf(path):
    """
    The UNIX command `rm -rf` for xcube.
    Recursively remove directory or single file.

    :param path:  directory or single file
    """
    import os
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
