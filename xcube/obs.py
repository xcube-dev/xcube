import s3fs
import xarray as xr
import zarr


def open_zarr(path: str, endpoint_url: str = None, max_cache_size: int = 2 ** 28) -> xr.Dataset:
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=endpoint_url))
    store = s3fs.S3Map(root=path, s3=s3, check=False)
    cached_store = zarr.LRUStoreCache(store, max_size=max_cache_size)
    return xr.open_zarr(cached_store)
