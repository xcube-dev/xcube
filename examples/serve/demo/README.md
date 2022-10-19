## About this xcube Server demo

### Running the xcube Server demo configuration

```shell
conda activate xcube
cd xcube
xcube serve --traceback --loglevel DETAIL serve --prefix api/v1 -vvv -c examples/serve/demo/config.yml
```

### Test data

The following data is used by the demo configuration `config.yml`:

* `cube.nc`: NetCDF data cube example

* `cube-1-250-250.zarr`: Zarr version of `cube.nc`.

* `cube-5-100-200.zarr`: Time-chunked version of `cube.nc`.

* `cube-1-250-250.levels`: Multi-level/multi-resolution (image pyramid) 
  version of `cube.nc`.

* `sample-geotiff.tif`: A simple GeoTIFF

* `sample-cog.tif`: A Cloud-optimized GeoTIFF with 3 overview levels 
  downloaded from
  https://rb.gy/h8qz14
  This image is one of many free GeoTIFFs available from 
  [Sentinel-2](https://registry.opendata.aws/sentinel-2-l2a-cogs/).

