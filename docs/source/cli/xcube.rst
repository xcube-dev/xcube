==========
``xcube``
==========

::

    $ xcube --help

::


    Usage: xcube [OPTIONS] COMMAND [ARGS]...

      xcube Toolkit

    Options:
      --version              Show the version and exit.
      --traceback            Enable tracing back errors by dumping the Python call
                             stack. Pass as very first option to also trace back
                             error during command-line validation.
      --scheduler SCHEDULER  Enable distributed computing using the Dask scheduler
                             identified by SCHEDULER. SCHEDULER can have the form
                             <address>?<keyword>=<value>,... where <address> is
                             <host> or <host>:<port> and specifies the scheduler's
                             address in your network. For more information on
                             distributed computing using Dask, refer to
                             http://distributed.dask.org/. Pairs of
                             <keyword>=<value> are passed to the Dask client.
                             Refer to http://distributed.dask.org/en/latest/api.ht
                             ml#distributed.Client
      --loglevel LOG_LEVEL   Log level. Must be one of OFF, CRITICAL, ERROR,
                             WARNING, INFO, DETAIL, DEBUG, TRACE. Defaults to OFF.
                             If the level is not OFF, any log messages up to the
                             given level will be written either to the console
                             (stderr) or LOG_FILE, if provided.
      --logfile LOG_FILE     Log file path. If given, any log messages will
                             redirected into LOG_FILE. Disables console output
                             unless otherwise enabled, e.g., using the --verbose
                             flag. Effective only if LOG_LEVEL is not OFF.
      -w, --warnings         Show any warnings emitted during operation (warnings
                             are hidden by default).
      --help                 Show this message and exit.

    Commands:
      chunk     (Re-)chunk xcube dataset.
      compute   Compute a cube from one or more other cubes.
      dump      Dump contents of an input dataset.
      extract   Extract cube points.
      gen       Generate xcube dataset.
      grid      Find spatial xcube dataset resolutions and adjust bounding...
      level     Generate multi-resolution levels.
      optimize  Optimize xcube dataset for faster access.
      patch     Patch and consolidate the metadata of a dataset.
      prune     Delete empty chunks.
      resample  Resample data along the time dimension.
      serve     Serve data cubes via web service.
      tile      IMPORTANT NOTE: The xcube tile tool in its current form is...
      vars2dim  Convert cube variables into new dimension.
      verify    Perform cube verification.
      versions  Get versions of important packages used by xcube.

