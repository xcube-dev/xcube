==================
``xcube optimize``
==================

Synopsis
========

Optimize xcube dataset for faster access.

::

    $ xcube optimize --help

::

    Usage: xcube optimize [OPTIONS] CUBE

      Optimize xcube dataset for faster access.

      Reduces the number of metadata and coordinate data files in xcube dataset
      given by CUBE. Consolidated cubes open much faster especially from remote
      locations, e.g. in object storage, because obviously much less HTTP requests
      are required to fetch initial cube meta information. That is, it merges all
      metadata files into a single top-level JSON file ".zmetadata". Optionally,
      it removes any chunking of coordinate variables so they comprise a single
      binary data file instead of one file per data chunk. The primary usage of
      this command is to optimize data cubes for cloud object storage. The command
      currently works only for data cubes using ZARR format.

    Options:
      -o, --output OUTPUT  Output path. The placeholder "<built-in function
                           input>" will be replaced by the input's filename
                           without extension (such as ".zarr"). Defaults to
                           "{input}-optimized.zarr".
      -I, --in-place       Optimize cube in place. Ignores output path.
      -C, --coords         Also optimize coordinate variables by converting any
                           chunked arrays into single, non-chunked, contiguous
                           arrays.
      --help               Show this message and exit.


Examples
========

Write an cube with consolidated metadata to ``cube-optimized.zarr``:

::

    $ xcube optimize ./cube.zarr
    
Write an optimized cube with consolidated metadata and consolidated coordinate variables to ``optimized/cube.zarr``
(directory ``optimized`` must exist):

::

    $ xcube optimize -C -o ./optimized/cube.zarr ./cube.zarr
    
Optimize a cube in-place with consolidated metadata and consolidated coordinate variables:

::

    $ xcube optimize -IC ./cube.zarr


Python API
==========

The related Python API function is :py:func:`xcube.core.optimize.optimize_dataset`.
