==================
``xcube resample``
==================

Synopsis
========

Resample data along the time dimension.

::

    $ xcube resample --help

::

    Usage: xcube resample [OPTIONS] CUBE

      Resample data along the time dimension.

    Options:
      -c, --config CONFIG             xcube dataset configuration file in YAML
                                      format. More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -o, --output OUTPUT             Output path. Defaults to 'out.zarr'.
      -f, --format [zarr|netcdf4|mem]
                                      Output format. If omitted, format will be
                                      guessed from output path.
      --variables, --vars VARIABLES   Comma-separated list of names of variables
                                      to be included.
      -M, --method TEXT               Temporal resampling method. Available
                                      downsampling methods are 'count', 'first',
                                      'last', 'min', 'max', 'sum', 'prod', 'mean',
                                      'median', 'std', 'var', the upsampling
                                      methods are 'asfreq', 'ffill', 'bfill',
                                      'pad', 'nearest', 'interpolate'. If the
                                      upsampling method is 'interpolate', the
                                      option '--kind' will be used, if given.
                                      Other upsampling methods that select
                                      existing values honour the '--tolerance'
                                      option. Defaults to 'mean'.
      -F, --frequency TEXT            Temporal aggregation frequency. Use format
                                      "<count><offset>" where <offset> is one of
                                      'H', 'D', 'W', 'M', 'Q', 'Y'. Use 'all' to
                                      aggregate all time steps included in the
                                      dataset.Defaults to '1D'.
      -O, --offset TEXT               Offset used to adjust the resampled time
                                      labels. Uses same syntax as frequency. Some
                                      Pandas date offset strings are supported as
                                      well.
      -B, --base INTEGER              For frequencies that evenly subdivide 1 day,
                                      the origin of the aggregated intervals. For
                                      example, for '24H' frequency, base could
                                      range from 0 through 23. Defaults to 0.
      -K, --kind TEXT                 Interpolation kind which will be used if
                                      upsampling method is 'interpolation'. May be
                                      one of 'zero', 'slinear', 'quadratic',
                                      'cubic', 'linear', 'nearest', 'previous',
                                      'next' where 'zero', 'slinear', 'quadratic',
                                      'cubic' refer to a spline interpolation of
                                      zeroth, first, second or third order;
                                      'previous' and 'next' simply return the
                                      previous or next value of the point. For
                                      more info refer to
                                      scipy.interpolate.interp1d(). Defaults to
                                      'linear'.
      -T, --tolerance TEXT            Tolerance for selective upsampling methods.
                                      Uses same syntax as frequency. If the time
                                      delta exceeds the tolerance, fill values
                                      (NaN) will be used. Defaults to the given
                                      frequency.
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
      --dry-run                       Just read and process inputs, but don't
                                      produce any outputs.
      --help                          Show this message and exit.

Examples
========

Upsampling example:

::

    $ xcube resample --vars conc_chl,conc_tsm -F 12H -T 6H -M interpolation -K linear examples/serve/demo/cube.nc

Downsampling example:

::

    $ xcube resample --vars conc_chl,conc_tsm -F 3D -M mean -M std -M count examples/serve/demo/cube.nc

Python API
==========

The related Python API function is :py:func:`xcube.core.resample.resample_in_time`.
