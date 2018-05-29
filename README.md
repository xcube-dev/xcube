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

# Tools

## `reproj-snap-nc`

    $ reproj-snap-nc -h
    usage: reproj-snap-nc [-h] [--version] [--dir OUTPUT_DIR] [--name OUTPUT_NAME]
                          [--format {nc,zarr}] [--size DST_SIZE]
                          [--region DST_REGION] [--variables DST_VARIABLES]
                          [--append]
                          INPUT_FILES [INPUT_FILES ...]
    
    Reproject SNAP NetCDF4 product
    
    positional arguments:
      INPUT_FILES           SNAP NetCDF4 products. May contain wildcards '?', '*',
                            and '**'.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version, -V         show program's version number and exit
      --dir OUTPUT_DIR, -d OUTPUT_DIR
                            Output directory. Defaults to '.'
      --name OUTPUT_NAME, -n OUTPUT_NAME
                            Output filename pattern. Defaults to
                            'PROJ_WGS84_{INPUT_FILE}'.
      --format {nc,zarr}, -f {nc,zarr}
                            Output format. Defaults to 'nc'.
      --size DST_SIZE, -s DST_SIZE
                            Output size in pixels using format "<width>,<height>".
                            Defaults to '512,512'.
      --region DST_REGION, -r DST_REGION
                            Output region using format "<lon-min>,<lat-min>,<lon-
                            max>,<lat-max>"
      --variables DST_VARIABLES, -v DST_VARIABLES
                            Variables to be included in output. Comma-separated
                            list of names.
      --append, -a          Append successive outputs.


Example:

    $ reproj-snap-nc -a -s 2000,1000 -r 0,50,5,52.5 -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -n hiroc-cube D:\OneDrive\BC\EOData\HIGHROC\2017\01\*.nc


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
