[![Build Status](https://travis-ci.com/dcs4cop/xcube.svg?branch=master)](https://travis-ci.com/dcs4cop/xcube)
[![codecov](https://codecov.io/gh/dcs4cop/xcube/branch/master/graph/badge.svg)](https://codecov.io/gh/dcs4cop/xcube)




# xcube

Data cubes with xarray

# Installation

First
    
    $ git clone https://github.com/bcdev/xcube.git
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


# Tools

## `xcube-genl2c`

    $ xcube-genl2c --help
    usage: xcube-genl2c [-h] [--version]
                        [--proc {rbins-seviri-highroc-scene-l2,rbins-seviri-highroc-daily-l2,snap-olci-highroc-l2,snap-olci-cyanoalert-l2}]
                        [--config CONFIG_FILE] [--dir OUTPUT_DIR]
                        [--name OUTPUT_NAME] [--writer {mem,netcdf4,zarr}]
                        [--size OUTPUT_SIZE] [--region OUTPUT_REGION]
                        [--vars OUTPUT_VARIABLES]
                        [--resamp {Nearest,Bilinear,Cubic,CubicSpline,Lanczos,Average,Min,Max,Median,Mode,Q1,Q3}]
                        [--traceback] [--append] [--dry-run]
                        INPUT_FILES [INPUT_FILES ...]
    
    Generate or extend a Level-2C data cube from Level-2 input files. Level-2C
    data cubes may be created in one go or in successively in append mode, input
    by input.
    
    positional arguments:
      INPUT_FILES           One or more input files or a pattern that may contain
                            wildcards '?', '*', and '**'.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version, -V         show program's version number and exit
      --proc {rbins-seviri-highroc-scene-l2,rbins-seviri-highroc-daily-l2,snap-olci-highroc-l2,snap-olci-cyanoalert-l2}, -p {rbins-seviri-highroc-scene-l2,rbins-seviri-highroc-daily-l2,snap-olci-highroc-l2,snap-olci-cyanoalert-l2}
                            Input processor type name.
      --config CONFIG_FILE, -c CONFIG_FILE
                            Data cube configuration file in YAML format.
      --dir OUTPUT_DIR, -d OUTPUT_DIR
                            Output directory. Defaults to '.'
      --name OUTPUT_NAME, -n OUTPUT_NAME
                            Output filename pattern. Defaults to
                            'PROJ_WGS84_{INPUT_FILE}'.
      --writer {mem,netcdf4,zarr}, -w {mem,netcdf4,zarr}
                            Output writer type name. Defaults to 'nc'.
      --size OUTPUT_SIZE, -s OUTPUT_SIZE
                            Output size in pixels using format "<width>,<height>".
      --region OUTPUT_REGION, -r OUTPUT_REGION
                            Output region using format "<lon-min>,<lat-min>,<lon-
                            max>,<lat-max>"
      --vars OUTPUT_VARIABLES, -v OUTPUT_VARIABLES
                            Variables to be included in output. Comma-separated
                            list of names which may contain wildcard characters
                            "*" and "?".
      --resamp {Nearest,Bilinear,Cubic,CubicSpline,Lanczos,Average,Min,Max,Median,Mode,Q1,Q3}
                            Fallback spatial resampling algorithm to be used for
                            all variables.Defaults to 'Nearest'.
      --traceback           On error, print Python traceback.
      --append, -a          Append successive outputs.
      --dry-run             Just read and process inputs, but don't produce any
                            outputs.
    
    input processors to be used with option --proc:
      rbins-seviri-highroc-scene-l2RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs
      rbins-seviri-highroc-daily-l2RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2        SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2     SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
    
    output formats to be used with option --writer:
      mem                     (*.mem)       In-memory dataset I/O
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)



Example:

    $ xcube-genl2c -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube -t snap-c2rcc D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc


## `xcube-genl3`

    $ xcube-genl3 --help
    usage: xcube-genl3 [-h] [--version] [--dir OUTPUT_DIR] [--name OUTPUT_NAME]
                       [--format {zarr,nc}] [--variables OUTPUT_VARIABLES]
                       [--resampling {all,any,argmin,argmax,count,first,last,max,mean,median,min,backfill,bfill,ffill,interpolate,nearest,pad}]
                       [--frequency OUTPUT_FREQUENCY]
                       [--meta-file OUTPUT_META_FILE] [--dry-run] [--traceback]
                       INPUT_FILE
    
    Generate Level-3 data cube from Level-2C data cube.
    
    positional arguments:
      INPUT_FILE            The input file or directory which must be a Level-2C
                            cube.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version, -V         show program's version number and exit
      --dir OUTPUT_DIR, -d OUTPUT_DIR
                            Output directory. Defaults to '.'
      --name OUTPUT_NAME, -n OUTPUT_NAME
                            Output filename pattern. Defaults to
                            'L3_{INPUT_FILE}'.
      --format {zarr,nc}, -f {zarr,nc}
                            Output format. Defaults to 'zarr'.
      --variables OUTPUT_VARIABLES, -v OUTPUT_VARIABLES
                            Variables to be included in output. Comma-separated
                            list of names which may contain wildcard characters
                            "*" and "?".
      --resampling {all,any,argmin,argmax,count,first,last,max,mean,median,min,backfill,bfill,ffill,interpolate,nearest,pad}
                            Temporal resampling method. Use format
                            "<count><offset>"where <offset> is one of {H, D, W, M,
                            Q, Y}Defaults to 'nearest'.
      --frequency OUTPUT_FREQUENCY
                            Temporal aggregation frequency.Defaults to '1D'.
      --meta-file OUTPUT_META_FILE, -m OUTPUT_META_FILE
                            File containing cube-level, CF-compliant metadata in
                            YAML format.
      --dry-run             Just read and process inputs, but don't produce any
                            outputs.
      --traceback           On error, print Python traceback.


## xcube-grid

    $ xcube-grid --help
    Usage: xcube-grid [OPTIONS] COMMAND [ARGS]...
    
      The Xcube grid tool is used to find suitable spatial data cube resolutions
      and to adjust bounding boxes to that resolutions.
    
      We find resolutions with respect to a fixed Earth grid and adjust regional
      spatial subsets to that fixed Earth grid. We also try to select the
      resolutions such that they are taken from a certain level of a multi-
      resolution pyramid whose level resolutions increase by a factor of two.
    
      The graticule on the fixed Earth grid is given by
    
          LON(I) = -180 + I * TILE / INV_RES
          LAT(J) =  -90 + J * TILE / INV_RES
    
      With
    
          INV_RES:  An integer number greater zero.
          RES:      1 / INV_RES, the spatial grid resolution in degrees.
          TILE:     Number of grid cells of a global grid at lowest resolution level.
    
      Let WIDTH and HEIGHT be the number of horizontal and vertical grid cells
      of a global grid at a certain LEVEL with WIDTH * RES = 360 and HEIGHT *
      RES = 180, then we also force HEIGHT = TILE * 2 ^ LEVEL.
    
    Options:
      --version  Show the version and exit.
      --help     Show this message and exit.
    
    Commands:
      abox    Adjust a bounding box to a fixed Earth grid.
      levels  List levels for target resolution.
      res     List resolutions close to a target...
    
Example: Find suitable target resolution for a ~300m (Sentinel 3 OLCI FR resolution) 
fixed Earth grid within a deviation of 5%.

    $ xcube-grid res 300m -d 5%
    
    TILE    LEVEL   HEIGHT  INV_RES RES (deg)       RES (m), DELTA_RES (%)
    540     7       69120   384     0.0026041666666666665   289.9   -3.4
    4140    4       66240   368     0.002717391304347826    302.5   0.8
    8100    3       64800   360     0.002777777777777778    309.2   3.1
    ...
    
289.9m is close enough and provides 7 resolution levels, which is good. Its inverse resolution is 384,
which is the fixed Earth grid identifier.

We want to see if the resolution pyramid also supports a resolution close to 10m 
(Sentinel 2 MSI resolution).

    $ xcube-grid levels 384 -m 6
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

    $ xcube-grid abox  0,50,5,52.5  384
     
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
