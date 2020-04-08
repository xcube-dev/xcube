## Changes in 0.5.0.dev0 (in development)

## Changes in 0.4.0

### New

* Added new `/timeseries/{dataset}/{variable}` POST operation to xcube web API.
  It extracts time-series for a given GeoJSON object provided as body.
  It replaces all of the `/ts/{dataset}/{variable}/{geom-type}` operations.
  The latter are still maintained for compatibility with the "VITO viewer". 
  
* The xcube web API provided through `xcube serve` can now serve RGBA tiles using the 
  `dataset/{dataset}/rgb/tiles/{z}/{y}/{x}` operation. The red, green, blue 
  channels are computed from three configurable variables and normalisation ranges, 
  the alpha channel provides transparency for missing values. To specify a default
  RGB schema for a dataset, a colour mapping for the "pseudo-variable" named `rbg` 
  is provided in the configuration of `xcube serve`:
  ```yaml  
  Datasets:
    - Identifyer: my_dataset
      Style: my_style
      ...
    ...
  Styles:
    - Identifier: my_style
      ColorMappings:
        rgb:
          Red:
            Variable: rtoa_8
            ValueRange: [0., 0.25]
          Green:
            Variable: rtoa_6
            ValueRange: [0., 0.25]
          Blue:
            Variable: rtoa_4
            ValueRange: [0., 0.25]
        ...
  ```
  Note that this concept works nicely in conjunction with the new `Augmentation` feature (#272) used
  to compute new variables that could be input to the RGB generation. 
  
* Introduced new (ortho-)rectification algorithm allowing reprojection of 
  satellite images that come with (terrain-corrected) geo-locations for every pixel.

  - new CLI tool `xcube rectify`
  - new API function `xcube.core.rectify.rectify_dataset()`

* Utilizing the new rectification in `xcube gen` tool. It is now the default 
  reprojection method in `xcube.core.gen.iproc.XYInputProcessor` and
  `xcube.core.gen.iproc.DefaultInputProcessor`, if ground control points are not 
  specified, i.e. the input processor is configured with `xy_gcp_step=None`. (#206)
* Tile sizes for rectification in `xcube gen` are now derived from `output_writer_params` if given in configuration and 
  if it contains a `chunksizes` parameter for 'lat' or 'lon'. This will force the generation of a chunked xcube dataset 
  and will utilize Dask arrays for out-of-core computations. This is very useful for large data cubes whose time slices 
  would otherwise not fit into memory.
* Introduced new function `xcube.core.select.select_spatial_subset()`.

* Renamed function `xcube.core.select.select_vars()` into `xcube.core.select.select_variables_subset()`.
  
* Now supporting xarray and numpy functions in expressions used by the
  `xcube.core.evaluate.evaluate_dataset()` function and in the configuration of the 
  `xcube gen` tool. You can now use `xr` and `np` contexts in expressions, e.g. 
  `xr.where(CHL >= 0.0, CHL)`. (#257)

* The performance of the `xcube gen` tool for the case that expressions or 
  expression parts are reused across multiple variables can now be improved. 
  Such as expressions can now be assigned to intermediate variables and loaded 
  into memory, so they are not recomputed again.
  For example, let the expression `quality_flags.cloudy and CHL > 25.0` occur often
  in the configuration, then this is how recomputation can be avoided:
  ```
    processed_variables:
      no_cloud_risk:
        expression: not (quality_flags.cloudy and CHL_raw > 25.0)
        load: True
      CHL:
        expression: CHL_raw
        valid_pixel_expression: no_cloud_risk
      ...        
  ```      
* Added ability to write xcube datasets in Zarr format into object storage bucket using the xcube python api
  `xcube.core.dsio.write_cube()`. (#224) The user needs to pass provide user credentials via 
  ```
  client_kwargs = {'provider_access_key_id': 'user_id', 'provider_secret_access_key': 'user_secret'}
  ```
  and 
  write to existing bucket by executing 
  
  ```
  write_cube(ds1, 'https://s3.amazonaws.com/upload_bucket/cube-1-250-250.zarr', 'zarr',
                       client_kwargs=client_kwargs)
  ```
* Added new CLI tool `xcube tile` which is used to generate a tiled RGB image 
  pyramid from any xcube dataset. The format and file organisation of the generated 
  tile sets conforms to the [TMS 1.0 Specification](https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification) 
  (#209).

* The configuration of `xcube serve` has been enhanced to support
  augmentation of data cubes by new variables computed on-the-fly (#272).
  You can now add a section `Augmentation` into a dataset descriptor, e.g.:
  
  ```yaml 
    Datasets:
      - Identifier: abc
        ...
        Augmentation:
          Path: compute_new_vars.py
          Function: compute_variables
          InputParameters:
            ...
      - ...
  ```
  
  where `compute_variables` is a function that receives the parent xcube dataset
  and is expected to return a new dataset with new variables. 
  
* The `xcube serve` tool now provides basic access control via OAuth2 bearer tokens (#263).
  To configure a service instance with access control, add the following to the 
  `xcube serve` configuration file:
  
  ```
    Authentication:
      Domain: "<your oauth2 domain>"
      Audience: "<your audience or API identifier>"
  ```
  
  Individual datasets can now be protected using the new `AccessControl` entry
  by configuring the `RequiredScopes` entry whose value is a list
  of required scopes, e.g. "read:datasets":
  
  ```
    Datasets:
      ...
      - Identifier: <some dataset id>
        ...
        AccessControl:
          RequiredScopes:
            - "read:datasets"
  ```
  
  If you want a dataset to disappear for authorized requests, set the 
  `IsSubstitute` flag:
  
  ```
    Datasets:
      ...
      - Identifier: <some dataset id>
        ...
        AccessControl:
          IsSubstitute: true
  ```

### Enhancements

* The `xcube serve` tool now also allows for per-dataset configuration
  of *chunk caches* for datasets read from remote object storage locations. 
  Chunk caching avoids recurring fetching of remote data chunks for same
  region of interest.
  It can be configured as default for all remote datasets at top-level of 
  the configuration file:
  ```
  DatasetChunkCacheSize: 100M
  ```
  or in individual dataset definitions:
  ```
  Datasets: 
     - Identifier: ...
       ChunkCacheSize: 2G
       ...
  ```
* Retrieval of time series in Python API function `xcube.core.timeseries.get_time_series()` 
  has been optimized and is now much faster for point geometries. 
  This enhances time-series performance of `xcube serve`. 
  * The log-output of `xcube serve` now contains some more details time-series request 
    so performance bottlenecks can be identified more easily from `xcube-serve.log`, 
    if the server is started together with the flag `--traceperf`.
* CLI command `xcube resample` has been enhanced by a new value for the 
  frequency option `--frequency all`
  With this value it will be possible to create mean, max , std, ... of the whole dataset,
  in other words, create an overview of a cube. 
  By [Alberto S. Rabaneda](https://github.com/rabaneda).
 
* The `xcube serve` tool now also serves dataset attribution information which will be 
  displayed in the xcube-viewer's map. To add attribution information, use the `DatasetAttribution` 
  in to your `xcube serve` configuration. It can be used on top-level (for all dataset), 
  or on individual datasets. Its value may be a single text entry or a list of texts:
  For example: 
  ```yaml
  DatasetAttribution: 
    - "© by Brockmann Consult GmbH 2020, contains modified Copernicus Data 2019, processed by ESA."
    - "Funded by EU H2020 DCS4COP project."
  ```
* The `xcube gen` tool now always produces consolidated xcube datasets when the output format is zarr. 
  Furthermore when appending to an existing zarr xcube dataset, the output now will be consolidated as well. 
  In addition, `xcube gen` can now append input time slices to existing optimized (consolidated) zarr xcube datasets.
* The `unchunk_coords` keyword argument of Python API function 
  `xcube.core.optimize.optimize_dataset()` can now be a name, or list of names  
  of the coordinate variable(s) to be consolidated. If boolean ``True`` is used
  all variables will be consolidated.
* The `xcube serve` API operations `datasets/` and `datasets/{ds_id}` now also
  return the metadata attributes of a given dataset and it variables in a property
  named `attrs`. For variables we added a new metadata property `htmlRepr` that is
  a string returned by a variable's `var.data._repr_html_()` method, if any.
* Renamed default log file for `xcube serve` command to `xcube-serve.log`.
* `xcube gen` now immediately flushes logging output to standard out
  
## Changes in 0.3.1 

### Fixes

* Removing false user warning about custom SNAP colormaps when starting 
  `xcube serve` command.
  
## Changes in 0.3.0

### New

* Added new parameter in `xcube gen` called `--no_sort`. Using `--no_sort`, 
  the input file list wont be sorted before creating the xcube dataset. 
  If `--no_sort` parameter is passed, order the input list will be kept. 
  The parameter `--sort` is deprecated and the input files will be sorted 
  by default. 
* xcube now discovers plugin modules by module naming convention
  and by Setuptools entry points. See new chapter 
  [Plugins](https://xcube.readthedocs.io/en/latest/plugins.html) 
  in xcube's documentation for details. (#211)  
* Added new `xcube compute` CLI command and `xcube.core.compute.compute_cube()` API 
  function that can be used to generate an output cube computed from a Python
  function that is applied to one or more input cubes. Replaces the formerly 
  hidden `xcube apply` command. (#167) 
* Added new function `xcube.core.geom.rasterize_features()` 
  to rasterize vector-data features into a dataset. (#222)
* Extended CLI command `xcube verify` and API function `xcube.core.verify.verify_cube` to check whether spatial
  coordinate variables and their associated bounds variables are equidistant. (#231)
* Made xarray version 0.14.1 minimum requirement due to deprecation of xarray's `Dataset.drop`
  method and replaced it with `drop_sel` and `drop_vars` accordingly. 


### Enhancements

* CLI commands execute much faster now when invoked with the `--help` and `--info` options.
* Added `serverPID` property to response of web API info handler. 
* Functions and classes exported by following modules no longer require data cubes to use
  the `lon` and `lat` coordinate variables, i.e. using WGS84 CRS coordinates. Instead, the 
  coordinates' CRS may be a projected coordinate system and coordinate variables may be called
  `x` and `y` (#112):
  - `xcube.core.new`
  - `xcube.core.geom`
  - `xcube.core.schema`
  - `xcube.core.verify`
* Sometimes the cell bounds coordinate variables of a given coordinate variables are not in a proper, 
  [CF compliant](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#cell-boundaries) 
  order, e.g. for decreasing latitudes `lat` the respective bounds coordinate
  `lat_bnds` is decreasing for `lat_bnds[:, 0]` and `lat_bnds[:, 1]`, but `lat_bnds[i, 0] < lat_bnds[i, 1]`
  for all `i`. xcube is now more tolerant w.r.t. to such wrong ordering of cell boundaries and will 
  compute the correct spatial extent. (#233)
* For `xcube serve`, any undefined color bar name will default to `"viridis"`. (#238)
    
 
### Fixes

* `xcube resample` now correctly re-chunks its output. By default, chunking of the 
  `time` dimension is set to one. (#212)

### Incompatible changes

The following changes introduce incompatibilities with former xcube 0.2.x 
versions. 

* The function specified by `xcube_plugins` entry points now receives an single argument of 
  type `xcube.api.ExtensionRegistry`. Plugins are asked to add their extensions
  to this registry. As an example, have a look at the default `xcube_plugins` entry points 
  in `./setup.py`.   
 
* `xcube.api.compute_dataset()` function has been renamed to 
  `xcube.api.evaluate_dataset()`. This has been done in order avoid confusion
  with new API function `xcube.api.compute_cube()`.
  
* xcube's package structure has been drastically changed: 
  - all of xcube's `__init__.py` files are now empty and no longer 
    have side effects such as sub-module aggregations. 
    Therefore, components need to be imported from individual modules.
  - renamed `xcube.api` into `xcube.core`
  - moved several modules from `xcube.util` into `xcube.core`
  - the new `xcube.constants` module contains package level constants
  - the new `xcube.plugin` module now registers all standard extensions
  - moved contents of module `xcube.api.readwrite` into `xcube.core.dsio`.
  - removed functions `read_cube` and `read_dataset` as `open_cube` and `open_dataset` are sufficient
  - all internal module imports are now absolute, rather than relative  

## Changes in 0.2.1

### Enhancements

- Added new CLI tool `xcube edit` and API function `xcube.api.edit_metadata`
  which allows editing the metadata of an existing xcube dataset. (#170)
- `xcube serve` now recognises xcube datasets with
  metadata consolidated by the `xcube opmimize` command. (#141)

### Fixes
- `xcube gen` now parses time stamps correcly from input data. (#207)
- Dataset multi-resolution pyramids (`*.levels` directories) can be stored in cloud object storage and are now usable with `xcube serve` (#179)
- `xcube optimize` now consolidates metadata only after consolidating
  coordinate variables. (#194)
- Removed broken links from `./README.md` (#197)
- Removed obsolete entry points from `./setup.py`.

## Changes in 0.2.0

### New

* Added first version of the [xcube documentation](https://xcube.readthedocs.io/) generated from `./docs` folder.

### Enhancements

* Reorganisation of the Documentation and Examples Section (partly addressing #106)
* Loosened python conda environment to satisfy conda-forge requirements
* xcube is now available as a conda package on the conda-forge channel. To install latest xcube package, you can now type: `conda install -c conda-forge xcube`
* Changed the unittesting code to minimize warnings reported by 3rd-party packages
* Making CLI parameters consistent and removing or changing parameter abbreviations in case they were used twice for different params. (partly addressing #91)
  For every CLI command which is generating an output a path must be provided by the option `-o`, `--output`. If not provided by the user, a default output_path is generated.
  The following CLI parameter have changed and their abbreviation is not enabled anymore : 

    - `xcube gen -v` is now only `xcube gen --vars` or `xcube gen --variables` 
    - `xcube gen -p` is now  `xcube gen -P` 
    - `xcube gen -i` is now  `xcube gen -I` 
    - `xcube gen -r` is now  `xcube gen -R`
    - `xcube gen -s` is now  `xcube gen -S` 
    - `xcube chunk -c`  is now  `xcube chunk -C`
    - `xcube level -l` is now `xcube level -L`
    - `xcube dump -v` is now `xcube dump --variable` or `xcube dump --var`
    - `xcube dump -e` is now `xcube dump -E` 
    - `xcube vars2dim -v` is now `xcube vars2dim --variable` or `xcube vars2dim --var`
    - `xcube vars2dim --var_name` is now `xcube vars2dim --variable` or `xcube vars2dim --var`
    - `xcube vars2dim -d` is now `xcube vars2dim -D` 
    - `xcube grid res -d` is now `xcube grid res -D`
    - `xcube grid res -c` is now `xcube grid res --cov` or `xcube grid res --coverage` 
    - `xcube grid res -n` is now `xcube grid res -N` or `xcube grid res --num_results` 
    - `xcube serve -p` is now `xcube serve -P` 
    - `xcube serve -a` is now `xcube serve -A` 
    
* Added option `inclStDev` and `inclCount` query parameters to `ts/{dataset}/{variable}/geometry` and derivates.
  If used with `inclStDev=1`, Xcube Viewer will show error bars for each time series point.
* `xcube.api.new_cube` function now accepts callables as values for variables.
  This allows to compute variable values depending on the (t, y, x) position
  in the cube. Useful for testing.
* `xcube.api` now exports the `MaskSet` class which is useful for decoding flag values encoding following the
  [CF conventions](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#flags).
* Added new CLI tool `xcube optimize` and API function `xcube.api.optimize_dataset` 
  optimizes data cubes for cloud object storage deployment. (#141)
* Added two new spatial dataset operations to Python API `xcube.api` (#148):
  * `mask_dataset_by_geometry(dataset, geometry)` clip and mask a dataset by geometry
  * `clip_dataset_by_geometry(dataset, geometry)` just clip a dataset by geometry 
* Changed the dev version tag from 0.2.0.dev3 to 0.2.0.dev
* The behavior of web API `/datasets?details=1` has changed.
  The call no longer includes associated vector data as GeoJSON. Instead new API
  has beed added to fetch new vector data on demand:
  `/datasets/{dataset}/places` and `/datasets/{dataset}/places/{place}` (#130)
* `xcube serve` accepts custom SNAP colormaps. The path to a SAP .cpd file can be passed via the server
   configuration file with the paramter [ColorFile] instead of [ColorBar]. (#84)
* `xcube serve` can now be configured to serve cubes that are associated 
   with another cube with same data but different chunking (#115). 
   E.g. using chunks such as `time=512,lat=1,lon=1` can drastically improve 
   time-series extractions. 
   Have a look at the demo config in `xube/webapi/res/demo/config.yml`.     
* `xcube serve` does now offer a AWS S3 compatible data access API (#97):
   - List bucket objects: `/s3bucket`, see AWS 
     docs [GET](https://docs.aws.amazon.com/AmazonS3/latest/API/v2-RESTBucketGET.html)
   - Get bucket object: `/s3bucket/{ds_id}/{path}`, 
     see AWS docs [HEAD](https://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectHEAD.html) 
     and [GET](https://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectGET.html)
* `xcube serve` now verifies that a configured cube is valid once it is opened. (#107)
* Added new CLI command `xcube verify` performing xcube dataset verification. (#19)
* Reworked `xcube extract` to be finally useful and effective for point data extraction. (#102) 
* `xcube server`can now filter datasets by point coordinate, e.g. `/datasets?point=12.5,52.8`. (#50) 
* `xcube server`can now limit time series to a maximum number of 
  valid (not NaN) values. To activate, pass optional query parameter `maxValids` to the various `/ts`
  functions. The special value `-1` will restrict the result to contain only valid values. (#113) 
* Reworked `xcube gen` to be more user-friendly and more consistent with other tools. 
  The changes are
  - Removed `--dir` and `--name` options and replaced it by single `--output` option, 
    whose default value is `out.zarr`. (#45)
  - The `--format` option no longer has a default value. If not given, 
    format is guessed from `--output` option.
  - Renamed following parameters in the configuration file:
    + `input_files` into `input_paths`, also because paths may point into object storage 
      locations (buckets);  
    + `output_file` into `output_path`, to be consistent with `input_paths`.  
* Added new CLI command `xcube prune`. The tool deletes all block files associated with empty (NaN-
  only) chunks in given INPUT cube, which must have ZARR format. This can drastically reduce files 
  in sparse cubes and improve cube reading performance. (#92)
* `xcube serve` has a new `prefix` option which is a path appended to the server's host.
  The `prefix` option replaces the `name` option which is now deprecated but kept 
  for backward compatibility. (#79)
* Added new CLI command `xcube resample` that is used to generate temporarily up- or downsampled
  data cubes from other data cubes.
* `xcube serve` can now be run with xcube dataset paths and styling information given via the CLI rather 
  than a configuration file. For example `xcube serve --styles conc_chl=(0,20,"viridis") /path/to/my/chl-cube.zarr`.
  This allows for quick inspection of newly generated cubes via `xcube gen`.
  Also added option `--show` that starts the Xcube viewer on desktop environments in a browser. 
* Added new `xcube apply` command that can be used to generate an output cube computed from a Python function 
  that is applied to one or more input cubes. 
  The command is still in development and therefore hidden.
* Added new `xcube timeit` command that can be used to measure the time required for 
  parameterized command invocations. 
  The command is still in development and therefore hidden.
* Added global `xcube --scheduler SCHEDULER` option for Dask distributed computing (#58)
* Added global `xcube --traceback` option, removed local `xcube gen --traceback` option
* Completed version 1 of an xcube developer guide.
* Added `xcube serve` command (#43) 
* `xcube serve`: Time-series web API now also returns "uncertainty" (#48)
* Added `xcube level` command to allow for creating spatial pyramid levels (#38)
* `xcube gen` accepts multiple configuration files that will be merged in order (#21)
* Added `xcube gen` option `--sort` when input data list should be sorted (#33)    
* Added `xcube vars2dim` command to make variables a cube dimension (#31)
* Added `xcube serve` option `--traceperf` that allows switching on performance diagnostics.
* Included possibility to read the input file paths from a text file. (#47)
* Restructured and clarified code base (#27)
* Moved to Python 3.7 (#25)
* Excluding all input processors except for the default one. They are now plugins and have own repositories within the 
xcube's organisation. (#49)


### Fixes

* `xcube gen` CLI now updates metadata correctly. (#181)
* It was no longer possible to use the `xcube gen` CLI with `--proc` option. (#120)
* `totalCount` attribute of time series returned by Web API `ts/{dataset}/{variable}/{geom-type}` now
   contains the correct number of possible observations. Was always `1` before.
* Renamed Web API function `ts/{dataset}/{variable}/places` into
  `ts/{dataset}/{variable}/features`.
* `xcube gen` is now taking care that when new time slices are added to an existing
   cube, this is done by maintaining the chronological order. New time slices are
   either appended, inserted, or replaced. (#64) (#139)
* Fixed `xcube serve` issue with WMTS KVP method `GetTile` with query parameter `time` 
  whose value can now also have the two forms `<start-date>/<end-date>` and just `<date>`. (#132) 
* Fixed `xcube extract` regression that stopped working after Pandas update (#95) 
* Fixed problem where CTRL+C didn't function anymore with `xcube serve`. (#87)
* Fixed error `indexes along dimension 'y' are not equal` occurred when using 
  `xcube gen` with processed variables that used flag values (#86)
* Fixed `xcube serve` WMTS KVP API to allow for case-insensitive query parameters. (#77)
* Fixed error in plugins when importing `xcube.api.gen` (#62)
* Fixed import of plugins only when executing `xcube.cli` (#66)

## Changes in 0.1.0

* Respecting chunk sizes when computing tile sizes [#44](https://github.com/dcs4cop/xcube-server/issues/44)
* The RESTful tile operations now have a query parameter `debug=1` which toggles tile 
  computation performance diagnostics.
* Can now associate place groups with datasets.
* Major revision of API. URLs are now more consistent.
* Request for obtaining a legend for a layer of given by a variable of a data set was added.
* Added a Dockerfile to build an xcube docker image and to run the demo
* The RESTful time-series API now returns ISO-formatted UTC dates [#26](https://github.com/dcs4cop/xcube-server/issues/26)
