[![Build Status](https://travis-ci.com/dcs4cop/xcube.svg?branch=master)](https://travis-ci.com/dcs4cop/xcube)
[![codecov](https://codecov.io/gh/dcs4cop/xcube/branch/master/graph/badge.svg)](https://codecov.io/gh/dcs4cop/xcube)




# xcube

Data cubes with xarray

# Installation

First
    
    $ git clone https://github.com/dcs4cop/xcube.git
    $ cd xcube
    $ conda env create
    
Then
    
    $ activate xcube-dev
    $ python setup.py develop

Update
    
    $ activate xcube-dev
    $ git pull --force
    $ python setup.py develop
    
    
Run tests

    $ pytest
    
with coverage

    $ pytest --cov=xcube

with [coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html) in HTML

    $ pytest --cov-report html --cov=xcube

# Developer Guide

...is [here](docs/DEV-GUIDE.md).


# Tools

## `xcube chunk`

    $ xcube chunk --help
    Usage: xcube chunk [OPTIONS] <input> <output>
    
    Write a new dataset with identical data and compression but with new chunk sizes.
    
    positional arguments:
      <input>           One data set in data cube format which needs to be chunked differently.
      <output>          Output name of rechunked data cube. 
    optional arguments:
      --help                Show this help message and exit
      --format, -f          Format of the output. If not given, guessed from <output>.
      --params, -p          Parameters specific for the output format. Comma-separated 
                            list of <key>=<value> pairs.
      --chunks, -c          Chunk sizes for each dimension. Comma-separated list
                            of <dim>=<size> pairs, e.g. "time=1,lat=270,lon=270"

Example:

    $ xcube chunk input_not_chunked.zarr output_rechunked.zarr --chunks "time=1,lat=270,lon=270"

## `xcube dump`

    $ xcube dump --help
    Usage: xcube dump [OPTIONS] <path>
    
    Dump contents of dataset.
    
    optional arguments:
      --help                Show this help message and exit
      --variable, -v        Name of a variable (multiple allowed).
      --encoding, -e        Dump also variable encoding information.


Example:

    $ xcube dump xcube_cube.zarr 

## `xcube extract`

    $ xcube dump --help
    Usage: xcube extract [OPTIONS] <cube> <coords>
    
    Extract data from <cube> at points given by coordinates <coords>.
    
    optional arguments:
    --help                 Show this message and exit.
    --indexes, -i          Include indexes in output.
    --output, -o <output>  Output file.
    --format, -f <format>  Format of the output. If not given, guessed from
                           <output>, otherwise <stdout> is used.
    --params <params>, -p  Parameters specific for the output format. Comma-
                           separated list of <key>=<value> pairs.

Example: # TODO: Help is needed here - how are the coords passed by the user? 
    
    $ xcube extract xcube_cube.zarr 

## xcube grid

    $ xcube grid --help
    Usage: xcube-grid [OPTIONS] COMMAND [ARGS]...
    
    Find suitable spatial data cube resolutions and to adjust bounding boxes
    to that resolutions.
    
    We find resolutions with respect to a possibly regional fixed Earth grid
    and adjust regional spatial subsets to that grid. We also try to select
    the resolutions such that they are taken from a certain level of a multi-
    resolution pyramid whose level resolutions increase by a factor of two.

    
    The graticule at a given resolution level L within the grid is given by
    
      RES(L) = COVERAGE * HEIGHT(L)
      HEIGHT(L) = HEIGHT_0 * 2 ^ L
      LON(L, I) = LON_MIN + I * HEIGHT_0 * RES(L)
      LAT(L, J) = LAT_MIN + J * HEIGHT_0 * RES(L)
    
    With
    
      RES:      Grid resolution in degrees.
      HEIGHT:   Number of vertical grid cells for given level
      HEIGHT_0: Number of vertical grid cells at lowest resolution level.

    
    Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
    of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT *
    RES = 180, then we also force HEIGHT = TILE * 2 ^ LEVEL.
    
    Options:
      --help     Show this message and exit.
    
    Commands:
      abox    Adjust a bounding box to a fixed Earth grid.
      levels  List levels for target resolution.
      res     List resolutions close to target resolution.
    
Example: Find suitable target resolution for a ~300m (Sentinel 3 OLCI FR resolution) 
fixed Earth grid within a deviation of 5%.

    $ xcube grid res 300m -d 5%
    
    TILE    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m), DELTA_RES (%)
    540     7       69120   384     0.0026041666666666665   289.9   -3.4
    4140    4       66240   368     0.002717391304347826    302.5   0.8
    8100    3       64800   360     0.002777777777777778    309.2   3.1
    ...
    
289.9m is close enough and provides 7 resolution levels, which is good. Its inverse resolution is 384,
which is the fixed Earth grid identifier.

We want to see if the resolution pyramid also supports a resolution close to 10m 
(Sentinel 2 MSI resolution).

    $ xcube grid levels 384 -m 6
    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m)
    0       540     3       0.3333333333333333      37106.5
    1       1080    6       0.16666666666666666     18553.2
    2       2160    12      0.08333333333333333     9276.6
    ...
    11      1105920 6144    0.00016276041666666666  18.1
    12      2211840 12288   8.138020833333333e-05   9.1
    13      4423680 24576   4.0690104166666664e-05  4.5

This indicates we have a resolution of 9.1m at level 12.

Lets assume we have data cube region with longitude from 0 to 5 degrees
and latitudes from 50 to 52.5 degrees. What is the adjusted bounding box 
on a fixed Earth grid with the inverse resolution 384?

    $ xcube grid abox  0,50,5,52.5  384
     
    Orig. box coord. = 0.0,50.0,5.0,52.5
    Adj. box coord.  = 0.0,49.21875,5.625,53.4375
    Orig. box WKT    = POLYGON ((0.0 50.0, 5.0 50.0, 5.0 52.5, 0.0 52.5, 0.0 50.0))
    Adj. box WKT     = POLYGON ((0.0 49.21875, 5.625 49.21875, 5.625 53.4375, 0.0 53.4375, 0.0 49.21875))
    Grid size  = 2160 x 1620 cells
    with
      TILE      = 540
      LEVEL     = 7
      INV_RES   = 384
      RES (deg) = 0.0026041666666666665
      RES (m)   = 289.89450727414993

    
Note, to check bounding box WKTs, you can use the 
handy tool [Wicket](https://arthur-e.github.io/Wicket/sandbox-gmaps3.html).
     
## `xcube gen`

    $ xcube gen --help
    usage: xcube gen [OPTIONS] INPUT_FILES
    
     Level-2C data cubes may be created in one go or successively in append
     mode, input by input. The input may be one or more input files or a
     pattern that may contain wildcards '?', '*', and '**'.
    
    optional arguments:
      --version                       Show the version and exit.
      -p, --proc INPUT_PROCESSOR      Input processor type name. The choices as
                                      input processor are: ['default', 'rbins-
                                      seviri-highroc-scene-l2', 'rbins-seviri-
                                      highroc-daily-l2', 'snap-olci-highroc-l2',
                                      'snap-olci-cyanoalert-l2'].  Additional
                                      information about input processors can be
                                      accessed by calling xcube generate_cube
                                      --info
      -c, --config CONFIG_FILE        Data cube configuration file in YAML format.
                                      More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -d, --dir OUTPUT_DIR            Output directory. Defaults to '.'
      -n, --name OUTPUT_NAME          Output filename pattern. Defaults to
                                      'PROJ_WGS84_{INPUT_FILE}'.
      -f, --format OUTPUT_FORMAT      Output writer type name. Defaults to 'zarr'.
                                      The choices for the output format are:
                                      ['csv', 'mem', 'netcdf4', 'zarr'].
                                      Additional information about output formats
                                      can be accessed by calling xcube
                                      generate_cube --info
      -s, --size OUTPUT_SIZE          Output size in pixels using format
                                      "<width>,<height>".
      -r, --region OUTPUT_REGION      Output region using format "<lon-min>,<lat-
                                      min>,<lon-max>,<lat-max>"
      -v, --variables OUTPUT_VARIABLES
                                      Variables to be included in output. Comma-
                                      separated list of names which may contain
                                      wildcard characters "*" and "?".
      --resampling OUTPUT_RESAMPLING  Fallback spatial resampling algorithm to be
                                      used for all variables. Defaults to
                                      'Nearest'. The choices for the resampling
                                      algorithm are: dict_keys(['Nearest',
                                      'Bilinear', 'Cubic', 'CubicSpline',
                                      'Lanczos', 'Average', 'Min', 'Max',
                                      'Median', 'Mode', 'Q1', 'Q3'])
      --traceback                     On error, print Python traceback.
      -a, --append                    Append successive outputs.
      --dry_run                       Just read and process inputs, but don't
                                      produce any outputs.
      -i, --info                      Displays additional information about format
                                      options or about input processors.
      --help                          Show this message and exit.

    $ xcube gen --info
    input processors to be used with option --proc:
      default                           Single-scene NetCDF/CF inputs in xcube standard format
      rbins-seviri-highroc-scene-l2     RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs
      rbins-seviri-highroc-daily-l2     RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2              SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2           SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
    
    
    output formats to be used with option --format:
      csv                     (*.csv)       CSV file format
      mem                     (*.mem)       In-memory dataset I/O
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)




Example:

    $ xcube gen -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube -t snap-c2rcc D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc


## `xcube level`

    $ xcube level --help

    Usage: xcube level [OPTIONS] <input>
    
      Transform the given dataset by <input> into the levels of a multi-level
      pyramid with spatial resolution decreasing by a factor of two in both
      spatial dimensions and write the result to directory <output>.
    
    Options:
      -o, --output <output>           Output directory. If omitted,
                                      "<input>.levels" will be used.
      -l, --link                      Link the <input> instead of converting it to
                                      a level zero dataset. Use with care, as the
                                      <input>'s internal spatial chunk sizes may
                                      be inappropriate for imaging purposes.
      -t, --tile-size <tile-size>     Tile size, given as single integer number or
                                      as <tile-width>,<tile-height>. If omitted,
                                      the tile size will be derived from the
                                      <input>'s internal spatial chunk sizes. If
                                      the <input> is not chunked, tile size will
                                      be 512.
      -n, --num-levels-max <num-levels-max>
                                      Maximum number of levels to generate. If not
                                      given, the number of levels will be derived
                                      from spatial dimension and tile sizes.
      --help                          Show this message and exit.

    
Example:

    $ xcube level -l -t 720 data/cubes/test-cube.zarr
