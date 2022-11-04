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

      Generate multi-resolution levels.

      Transform the given dataset by INPUT into the levels of a multi-level
      pyramid with spatial resolution decreasing by a factor of two in both
      spatial dimensions and write the result to directory OUTPUT.

      INPUT may be an S3 object storage URL of the form "s3://<bucket>/<path>" or
      "https://<endpoint>".

    Options:
      -o, --output OUTPUT             Output path. If omitted, "INPUT.levels" will
                                      be used. You can also use S3 object storage
                                      URLs of the form "s3://<bucket>/<path>" or
                                      "https://<endpoint>"
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
      -A, --agg-methods AGG_METHODS   Aggregation method(s) to be used for data
                                      variables. Either one of "first", "min",
                                      "max", "mean", "median", "auto" or list of
                                      assignments to individual variables using
                                      the notation
                                      "<var1>=<method1>,<var2>=<method2>,..."
                                      Defaults to "first".
      -r, --replace                   Whether to replace an existing dataset at
                                      OUTPUT.
      -a, --anon                      For S3 inputs or outputs, whether the access
                                      is anonymous. By default, credentials are
                                      required.
      -q, --quiet                     Disable output of log messages to the
                                      console entirely. Note, this will also
                                      suppress error and warning messages.
      -v, --verbose                   Enable output of log messages to the
                                      console. Has no effect if --quiet/-q is
                                      used. May be given multiple times to control
                                      the level of log messages, i.e., -v refers
                                      to level INFO, -vv to DETAIL, -vvv to DEBUG,
                                      -vvvv to TRACE. If omitted, the log level of
                                      the console is WARNING.
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
