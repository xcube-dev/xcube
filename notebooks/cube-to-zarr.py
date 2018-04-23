import cablab as cl
import xarray as xr
import zarr

print(xr.version.version)

cube = cl.Cube.open('D:\\EOData\\cablab-datacube-1.0.0\\low-res')

print('converting to xarray dataset')
ds = cube.data.dataset()

print('writing uncompressed zarr')
ds.to_zarr('D:\\EOData\\cablab-datacube-1.0.0.zarr')

print('writing compressed zarr')
compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
ds.to_zarr('D:\\EOData\\cablab-datacube-1.0.0.comp.zarr',
           encoding={'cablab-datacube-1.0.0.comp': {'compressor': compressor}})
