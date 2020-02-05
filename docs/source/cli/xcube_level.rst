===============
``xcube level``
===============

Synopsis
========

Generate multi-resolution levels.

::

    $ xcube level --help

::

    Usage: xcube level [OPTIONS] INPUT

      Generate multi-resolution levels. Transform the given dataset by INPUT
      into the levels of a multi-level pyramid with spatial resolution
      decreasing by a factor of two in both spatial dimensions and write the
      result to directory OUTPUT.

    Options:
      -o, --output OUTPUT             Output path. If omitted, "INPUT.levels" will
                                      be used.
      -L, --link                      Link the INPUT instead of converting it to a
                                      level zero dataset. Use with care, as the
                                      INPUT's internal spatial chunk sizes may be
                                      inappropriate for imaging purposes.
      -t, --tile-size TILE_SIZE       Tile size, given as single integer number or
                                      as <tile-width>,<tile-height>. If omitted,
                                      the tile size will be derived from the
                                      INPUT's internal spatial chunk sizes. If the
                                      INPUT is not chunked, tile size will be 512.
      -n, --num-levels-max NUM_LEVELS_MAX
                                      Maximum number of levels to generate. If not
                                      given, the number of levels will be derived
                                      from spatial dimension and tile sizes.
      --help                          Show this message and exit.



    
Example
=======

::

    $ xcube level --link -t 720 data/cubes/test-cube.zarr

Python API
==========


The related Python API functions are

* :py:func:`xcube.core.level.compute_levels`,
* :py:func:`xcube.core.level.read_levels`, and
* :py:func:`xcube.core.level.write_levels`.
