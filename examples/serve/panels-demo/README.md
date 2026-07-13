## About this xcube Server demo

### Running the xcube Server demo configuration

```shell
conda activate xcube
cd xcube
xcube serve --traceback --loglevel -vvv -c examples/serve/panels-demo/config.yaml
```

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

