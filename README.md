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
    usage: xcube-genl2c [-h] [--version] [--dir OUTPUT_DIR] [--name OUTPUT_NAME]
                        [--format {netcdf4,zarr}] [--size OUTPUT_SIZE]
                        [--region OUTPUT_REGION] [--meta-file OUTPUT_META_FILE]
                        [--variables OUTPUT_VARIABLES] [--append] [--dry-run]
                        [--type {snap-olci-highroc-l2}]
                        INPUT_FILES [INPUT_FILES ...]
    
    Generate L2C data cube from various input files. L2C data cubes may be created
    in one go or in successively in append mode, input by input.
    
    positional arguments:
      INPUT_FILES           One or more input files or a pattern that may contain
                            wildcards '?', '*', and '**'.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version, -V         show program's version number and exit
      --dir OUTPUT_DIR, -d OUTPUT_DIR
                            Output directory. Defaults to '.'
      --name OUTPUT_NAME, -n OUTPUT_NAME
                            Output filename pattern. Defaults to
                            'PROJ_WGS84_{INPUT_FILE}'.
      --format {netcdf4,zarr}, -f {netcdf4,zarr}
                            Output format. Defaults to 'netcdf4'.
      --size OUTPUT_SIZE, -s OUTPUT_SIZE
                            Output size in pixels using format "<width>,<height>".
                            Defaults to '512,512'.
      --region OUTPUT_REGION, -r OUTPUT_REGION
                            Output region using format "<lon-min>,<lat-min>,<lon-
                            max>,<lat-max>"
      --meta-file OUTPUT_META_FILE, -m OUTPUT_META_FILE
                            File containing cube-level CF-compliant metadata in
                            YAML format.
      --variables OUTPUT_VARIABLES, -v OUTPUT_VARIABLES
                            Variables to be included in output. Comma-separated
                            list of names.
      --append, -a          Append successive outputs.
      --dry-run             Just read and process inputs, but don't produce any
                            outputs.
      --type {snap-olci-highroc-l2}, -t {snap-olci-highroc-l2}
                            Input type. Defaults to 'snap-olci-highroc-l2'.
    
    output formats to be used with option --format:
      netcdf4                 (*.nc)        NetCDF-4 file format
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)
    
    input types to be used with option --type:
      snap-olci-highroc-l2    (*.nc)        SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs



Example:

    $ xcube-genl2c -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube -t snap-c2rcc D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc


# Cube generation notes

## Grid definition

The data cubes in the DCS4COP project are generated for predefined regions and predefined spatial resolutions.
A cube for a given region may be generated for multiple spatial resolutions.

There may be a scenario in which all the region cubes for will be used together. Then,
in order to avoid another spatial resampling to a common grid, the grids of all cubes should be
defined on a fixed Earth grid.

This means a cube's region coordinates are taken from the corresponding cells 
of a fixed Earth grid defined for a given spatial resolution.

Therefore the `num_levels` spatial resolutions in which data cubes are produced shall be defined such that
the fixed Earth's grid sizes multiplied by the cell size `delta` is close to 180 degrees.
In addition we want the actual data cubes sizes to be integer multiples of an appropriate internal
chunk (tile) size used to compress and store the data at different resolution levels `i`:

    |grid_size[i] * delta[i] - 180| < eps
    num_chunks[i] = num_chunks[0] * power(2, i)
    grid_size[i] = num_chunks[i] * chunk_size

with

    0 <= i < num_levels
    num_chunks[0] := number of chunks at lowest spatial resolution i in both spatial dimensions
    chunk_size := size of a chunk in both spatial dimensions


# HIGHROC sample data

* ftp://ftp.brockmann-consult.de/
* user: highrocre
* pw: H!Ghr0$1
