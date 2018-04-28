# xcube

Data cubes with xarray

# Installation

First time users:

    $ conda env create
    
Then

    $ activate xcube-dev
    $ python setup.py develop


# Tools

## `reproj-snap-nc`

    $ reproj-snap-nc -h
    usage: reproj-snap-nc [-h] [--version] [--dir OUTPUT_DIR]
                          [--pattern OUTPUT_PATTERN] [--format {nc,zarr}]
                          [--size DST_SIZE] [--region DST_REGION]
                          input_file

    Reproject SNAP NetCDF4 product

    positional arguments:
      input_file            SNAP NetCDF4 product

    optional arguments:
      -h, --help            show this help message and exit
      --version, -V         show program's version number and exit
      --dir OUTPUT_DIR, -d OUTPUT_DIR
                            Output directory
      --pattern OUTPUT_PATTERN, -p OUTPUT_PATTERN
                            Output filename pattern
      --format {nc,zarr}, -f {nc,zarr}
                            Output format
      --size DST_SIZE, -s DST_SIZE
                            Output size in pixels using format "<width>,<height>"
      --region DST_REGION, -r DST_REGION
                            Output region using format "<lon-min>,<lat-min>,<lon-
                            max>,<lat-max>"


# Cube generation notes

## Grid definition

DCS4COP cubes are generated for predefined regions and predefined spatial resolutions. 
A cube for a given region may be generated for multiple spatial resolutions.

There may be a scenario in which all the region cubes for will be used together. Then
it would be ideal, if the grids of the cubes are defined on a fixed Earth grid.

This means a cube's region coordinates are taken from the corresponding cells 
of a fixed Earth grid defined for a given spatial resolution.

Therefore the `N` spatial resolutions shall be defined such that each subdivide
180 degrees into an integer number with a sufficiently small remainder `eps`:

    |count[i] * delta[i] - 180| < eps, for 0 <= i < N
    
When given `count[0] = COUNT_MIN` at lowest resolution (with largest `delta[i]`) 
then

    count[i] = count[0] * f(i), for 1 < i < N
    
where for example `f(i) = 2 ^ i` is commonly used for image pyramids.


# Sample data

* ftp://ftp.brockmann-consult.de/
* user: highrocre
* pw: H!Ghr0$1