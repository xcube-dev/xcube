## About this xcube Server demo

### Running the xcube Server demo configuration

```shell
pixi run xcube serve -vvv -c examples/serve/panels-demo/config.yaml
```

or simpler

- `pixi run viewer-panels-demo` to run the xcube Server and open xcube Viewer in a browser tab
- `pixi run server-panels-demo` to run the xcube Server only


### Test data

The following data is used by the demo configuration `config.yaml`:

1. Kattegat
   - Derived from CMEMS product BALTICSEA_ANALYSISFORECAST_BGC_003_007
   - Spatial, temporal, variable and depth subset
   - Original dataset: https://data.marine.copernicus.eu/product/BALTICSEA_ANALYSISFORECAST_BGC_003_007/description

2. Waddensea
   - Derived from Copernicus Sentinel-2 Level-2A scenes
   - Spatial, temporal, scene and band subset

Both datasets are provided solely for demonstration purposes and contain modified Copernicus information.
