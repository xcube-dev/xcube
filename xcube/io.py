import glob
from abc import abstractmethod, ABCMeta
from typing import Set, Callable, List, Optional

import s3fs
import xarray as xr
import zarr


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
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def ext(self) -> str:
        pass

    @property
    @abstractmethod
    def modes(self) -> Set[str]:
        pass

    @abstractmethod
    def read(self, input_path: str, **kwargs) -> xr.Dataset:
        pass

    @abstractmethod
    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        pass

    @abstractmethod
    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        pass


class Netcdf4DatasetIO(DatasetIO):

    @property
    def name(self) -> str:
        return 'netcdf4'

    @property
    def description(self) -> str:
        return 'NetCDF-4 file format'

    @property
    def ext(self) -> str:
        return 'nc'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w', 'a'}

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
    def __init__(self):
        self.root_group = None

    @property
    def name(self) -> str:
        return 'zarr'

    @property
    def description(self) -> str:
        return 'Zarr file format (http://zarr.readthedocs.io)'

    @property
    def ext(self) -> str:
        return 'zarr'

    @property
    def modes(self) -> Set[str]:
        return {'r', 'w', 'a'}

    def read(self, path: str, **kwargs) -> xr.Dataset:
        return xr.open_zarr(path, **kwargs)

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        encoding = dict()
        for var_name in dataset.data_vars:
            new_var = dataset[var_name]
            encoding[var_name] = {'compressor': compressor, 'chunks': new_var.shape}
        dataset.to_zarr(output_path,
                        encoding=encoding)

    def append(self, dataset: xr.Dataset, output_path: str):
        import zarr
        if self.root_group is None:
            self.root_group = zarr.open(output_path, mode='a')
        for var_name, var_array in self.root_group.arrays():
            new_var = dataset[var_name]
            if 'time' in new_var.dims:
                axis = new_var.dims.index('time')
                var_array.append(new_var, axis=axis)


class OpendapDatasetIO(DatasetIO):

    @property
    def name(self) -> str:
        return 'opendap'

    @property
    def description(self) -> str:
        return 'OPeNDAP protocol'

    @property
    def ext(self) -> str:
        return 'nc'

    @property
    def modes(self) -> Set[str]:
        return {'r'}

    def read(self, path: str, **kwargs) -> xr.Dataset:
        return xr.open_dataset(path, engine='pydap', **kwargs)

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()


class DatasetIORegistry:

    def __init__(self, registrations):
        self._registrations = list(registrations)

    def register(self, dataset_io: DatasetIO):
        # noinspection PyTypeChecker
        self._registrations.append(dataset_io)

    def find(self, format_name: str, modes: Set[str] = None, default: DatasetIO = None) -> Optional[DatasetIO]:
        registrations = list(self._registrations)
        format_name = format_name.lower()
        for dataset_io in registrations:
            # noinspection PyUnresolvedReferences
            if format_name == dataset_io.name.lower():
                # noinspection PyTypeChecker
                if not modes or modes.issubset(dataset_io.modes):
                    return dataset_io
        for dataset_io in registrations:
            # noinspection PyUnresolvedReferences
            if format_name == dataset_io.ext.lower():
                # noinspection PyTypeChecker
                if not modes or modes.issubset(dataset_io.modes):
                    return dataset_io
        return default

    def query(self, filter_fn: Callable[[DatasetIO], bool] = None) -> List[DatasetIO]:
        registrations = list(self._registrations)
        if filter is None:
            return registrations
        return list(filter(filter_fn, registrations))


_DATASET_IO_REGISTRY = DatasetIORegistry([Netcdf4DatasetIO(), ZarrDatasetIO(), OpendapDatasetIO()])


def get_default_dataset_io_registry() -> DatasetIORegistry:
    return _DATASET_IO_REGISTRY


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
