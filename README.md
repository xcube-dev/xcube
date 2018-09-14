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


## `xcube-grid`

    $xcube-grid --help
    usage: xcube-grid [-h] [--units {degrees,deg,meters,m}] [--res RES]
                      [--bbox BBOX] [--size SIZE]
    
    Fixed Earth Grid Calculator
    
    optional arguments:
      -h, --help            show this help message and exit
      --units {degrees,deg,meters,m}, -u {degrees,deg,meters,m}
                            Coordinate units
      --res RES             Desired resolution in given units
      --bbox BBOX           Desired bounding box <xmin>,<ymin>,<xmax>,<ymax> in
                            given units
      --size SIZE           Desired spatial image size <width>,<height> in pixels
    
