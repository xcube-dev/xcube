# `xcube serve`

Serve data cubes via web service.

The `xcube serve` is a light-weight web server that provides various services based on
xcube data cubes:

* Catalogue services to query for datasets and their variables and dimensions, and feature collections.
* Tile map service, with some OGC WMTS 1.0 compatibility (REST and KVP APIs)
* Dataset services to extract subsets like time-series and profiles for e.g. JS clients

Find its API description [here](https://app.swaggerhub.com/apis-docs/bcdev/xcube-server).

xcube datasets are any datasets that

* that comply to [Unidata's CDM](https://www.unidata.ucar.edu/software/thredds/v4.3/netcdf-java/CDM/) and to the [CF Conventions](http://cfconventions.org/);
* that can be opened with the [xarray](https://xarray.pydata.org/en/stable/) Python library;
* that have variables that have at least the dimensions and shape (`time`, `lat`, `lon`), in exactly this order;
* that have 1D-coordinate variables corresponding to the dimensions;
* that have their spatial grid defined in the WGS84 (`EPSG:4326`) coordinate reference system.

The Xcube server supports local NetCDF files or local or remote [Zarr](https://zarr.readthedocs.io/en/stable/) directories.
Remote Zarr directories must be stored in publicly accessible, AWS S3 compatible
object storage (OBS).

As an example, here is the [configuration of the demo server](https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config.yml).

## OGC WMTS compatibility

The Xcube server implements the RESTful and KVP architectural styles
of the [OGC WMTS 1.0.0 specification](http://www.opengeospatial.org/standards/wmts).

The following operations are supported:

* **GetCapabilities**: `/xcube/wmts/1.0.0/WMTSCapabilities.xml`
* **GetTile**: `/xcube/wmts/1.0.0/tile/{DatasetName}/{VarName}/{TileMatrix}/{TileCol}/{TileRow}.png`
* **GetFeatureInfo**: *in progress*

## Explore API of existing xcube-servers

To explore the API of existing xcube-servers go to the [SwaggerHub of bcdev](https://app.swaggerhub.com/apis/bcdev/xcube-server/0.1.0.dev6).
The SwaggerHub allows to choose the xcube-server project and therefore the datasets which are used for the exploration.


```bash
$ xcube serve --help
```

    Usage: xcube serve [OPTIONS] [CUBE]...
    
      Serve data cubes via web service.
    
      Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP
      interface. The RESTful API documentation can be found at
      https://app.swaggerhub.com/apis/bcdev/xcube-server.
    
    Options:
      -A, --address ADDRESS  Service address. Defaults to 'localhost'.
      -P --port PORT            Port number where the service will listen on.
                             Defaults to 8080.
      --prefix PREFIX        Service URL prefix. May contain template patterns
                             such as "${version}" or "${name}". For example
                             "${name}/api/${version}".
      -u, --update PERIOD    Service will update after given seconds of
                             inactivity. Zero or a negative value will disable
                             update checks. Defaults to 2.0.
      -S, --styles STYLES    Color mapping styles for variables. Used only, if one
                             or more CUBE arguments are provided and CONFIG is not
                             given. Comma-separated list with elements of the form
                             <var>=(<vmin>,<vmax>) or
                             <var>=(<vmin>,<vmax>,"<cmap>")
      -c, --config CONFIG    Use datasets configuration file CONFIG. Cannot be
                             used if CUBES are provided.
      --tilecache SIZE       In-memory tile cache size in bytes. Unit suffixes
                             'K', 'M', 'G' may be used. Defaults to '512M'. The
                             special value 'OFF' disables tile caching.
      --tilemode MODE        Tile computation mode. This is an internal option
                             used to switch between different tile computation
                             implementations. Defaults to 0.
      -s, --show                 Run viewer app. Requires setting the environment
                             variable XCUBE_VIEWER_PATH to a valid xcube-viewer
                             deployment or build directory. Refer to
                             https://github.com/dcs4cop/xcube-viewer for more
                             information.
      -v, --verbose          Delegate logging to the console (stderr).
      --traceperf            Print performance diagnostics (stdout).
      --help                 Show this message and exit.

