# `xcube gen`

Generate data cube.

```bash
$ xcube gen --help
```

    Usage: xcube gen [OPTIONS] [INPUTS]...
    
      Generate data cube. Data cubes may be created in one go or successively in
      append mode, input by input. The input paths may be one or more input
      files or a pattern that may contain wildcards '?', '*', and '**'. The
      input paths can also be passed as lines of a text file. To do so, provide
      exactly one input file with ".txt" extension which contains the actual
      input paths to be used.
    
    Options:
      -p, --proc TEXT                 Input processor name. The available input
                                      processor names and additional information
                                      about input processors can be accessed by
                                      calling xcube gen --info . Defaults to
                                      "default", an input processor that can deal
                                      with simple datasets whose variables have
                                      dimensions ("lat", "lon") and conform with
                                      the CF conventions.
      -c, --config TEXT               Data cube configuration file in YAML format.
                                      More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -o, --output TEXT               Output path. Defaults to 'out.zarr'
      -f, --format TEXT               Output format. Information about output
                                      formats can be accessed by calling xcube gen
                                      --info. If omitted, the format will be
                                      guessed from the given output path.
      -s, --size TEXT                 Output size in pixels using format
                                      "<width>,<height>".
      -r, --region TEXT               Output region using format "<lon-min>,<lat-
                                      min>,<lon-max>,<lat-max>"
      -v, --variables, --vars TEXT    Variables to be included in output. Comma-
                                      separated list of names which may contain
                                      wildcard characters "*" and "?".
      --resampling [Average|Bilinear|Cubic|CubicSpline|Lanczos|Max|Median|Min|Mode|Nearest|Q1|Q3]
                                      Fallback spatial resampling algorithm to be
                                      used for all variables. Defaults to
                                      'Nearest'. The choices for the resampling
                                      algorithm are: ['Average', 'Bilinear',
                                      'Cubic', 'CubicSpline', 'Lanczos', 'Max',
                                      'Median', 'Min', 'Mode', 'Nearest', 'Q1',
                                      'Q3']
      -a, --append                    Deprecated. The command will now always
                                      create, insert, replace, or append input
                                      slices.
      --prof                          Collect profiling information and dump
                                      results after processing.
      --sort                          The input file list will be sorted before
                                      creating the data cube. If --sort parameter
                                      is not passed, order of input list will be
                                      kept.
      -i, --info                      Displays additional information about format
                                      options or about input processors.
      --dry_run                       Just read and process inputs, but don't
                                      produce any outputs.
      --help                          Show this message and exit.


Below is the `xcube gen --info` call with 5 input processors installed via plugins.

    $ xcube gen --info
    input processors to be used with option --proc:
      default                           Single-scene NetCDF/CF inputs in xcube standard format
      rbins-seviri-highroc-scene-l2     RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs
      rbins-seviri-highroc-daily-l2     RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2              SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2           SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
      vito-s2plus-l2                    VITO Sentinel-2 Plus Level 2 NetCDF inputs

    For more input processors use existing "xcube-gen-..." plugins from the github organisation DCS4COP or write own plugin.


    output formats to be used with option --format:
      csv                     (*.csv)       CSV file format
      mem                     (*.mem)       In-memory dataset I/O
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)


Example:

    $ xcube gen -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube -t -p default D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc


Some input processors have been developed for specific EO data sources 
used within the DCS4COP project. They may serve as examples how to develop
input processor plug-ins:

* `xcube-gen-rbins <https://github.com/dcs4cop/xcube-gen-rbins>`_
* `xcube-gen-bc <https://github.com/dcs4cop/xcube-gen-bc>`_
* `xcube-gen-vito <https://github.com/dcs4cop/xcube-gen-vito>`_

