## Changes in 0.2.0 (in dev)

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
* Fixed problem where CTRL+C didn't function anymore with `xcube serve`. (#87)
* Fixed error `indexes along dimension 'y' are not equal` occurred when using 
  `xcube gen` with processed variables that used flag values (#86)
* `xcube serve` has a new `prefix` option which is a path appended to the server's host.
  The `prefix` option replaces the `name` option which is now deprecated but kept 
  for backward compatibility. (#79)
* Fixed `xcube serve` WMTS KVP API to allow for case-insensitive query parameters. (#77)
* Added new CLI command `xcube resample` that is used to generate temporarily up- or 
  downsampled data cubes from other data cubes.
* `xcube serve` can now be run with data cube paths and styling information given via the CLI rather 
  than a configuration file. For example `xcube serve --styles conc_chl=(0,20,"viridis") /path/to/my/chl-cube.zarr`.
  This allows for quick inspection of newly generated cubes via `xcube gen`.
  Also added option `--show` that starts the Xcube viewer on desktop environments in a browser. 
* Added new `xcube apply` command that can be used to generate an output cube computed from a Python function 
  that is applied to one or more input cubes. 
  The command is still in development and therefore hidden.
* Added new `xcube timeit` command that can be used to measure the time required for 
  parameterized command invocations. 
  The command is still in development and therefore hidden.
* Added global `xcube --scheduler <scheduler>` option for Dask distributed computing (#58)
* Added global `xcube --traceback` option, removed local `xcube gen --traceback` option
* Completed version 1 of an xcube developer guide.
* Added `xcube serve` command (#43) 
* `xcube serve`: Time-series web API now also returns "uncertainty" (#48)
* Added `xcube level` command to allow for creating spatial pyramid levels (#38)
* `xcube gen` accepts multiple configuration files that will be merged in order (#21)
* Added `xcube gen` option `--sort` when input data list should be sorted (#33)    
* Added `xcube vars2dim` command to make variables a cube dimension (#31)
* Added `xcube serve` option `--traceperf` that allows switching on performance diagnostics.
* Fixed error in plugins when importing `xcube.api.gen` (#62)
* Fixed import of plugins only when executing `xcube.cli` (#66)
* Included possibility to read the input file paths from a text file. (#47)
* Restructured and clarified code base (#27)
* Moved to Python 3.7 (#25)
* Excluding all input processors except for the default one. 
  They are now plugins and have own repositories within the 
  xcube's organisation. (#49)

## Changes in 0.1.0

* Respecting chunk sizes when computing tile sizes [#44](https://github.com/dcs4cop/xcube-server/issues/44)
* The RESTful tile operations now have a query parameter `debug=1` which toggles tile 
  computation performance diagnostics.
* Can now associate place groups with datasets.
* Major revision of API. URLs are now more consistent.
* Request for obtaining a legend for a layer of given by a variable of a data set was added.
* Added a Dockerfile to build an xcube docker image and to run the demo
* The RESTful time-series API now returns ISO-formatted UTC dates [#26](https://github.com/dcs4cop/xcube-server/issues/26)

