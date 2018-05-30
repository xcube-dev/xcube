import s3fs
import xarray as xr
import zarr


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
