==========
``xcube``
==========

::

    $ xcube --help

::


    Usage: xcube [OPTIONS] COMMAND [ARGS]...
    
    Xcube Toolkit
    
    Options:
      --version                Show the version and exit.
      --traceback              Enable tracing back errors by dumping the Python
                               call stack. Pass as very first option to also trace
                               back error during command-line validation.
      --scheduler SCHEDULER    Enable distributed computing using the Dask
                               scheduler identified by SCHEDULER. SCHEDULER
                               can have the form <address>?<keyword>=<value>,...
                               where <address> is <host> or <host>:<port> and
                               specifies the scheduler's address in your network.
                               For more information on distributed computing using
                               Dask, refer to http://distributed.dask.org/. Pairs
                               of <keyword>=<value> are passed to the Dask client.
                               Refer to http://distributed.dask.org/en/latest/api.
                               html#distributed.Client
      --help                   Show this message and exit.
    
    Commands:
      chunk     (Re-)chunk xcube dataset.
      compute   Compute a cube from one or more other cubes.
      dump      Dump contents of an input dataset.
      edit      Edit the metadata of an xcube dataset.
      extract   Extract cube points.
      gen       Generate xcube dataset.
      grid      Find spatial xcube dataset resolutions and adjust bounding...
      level     Generate multi-resolution levels.
      optimize  Optimize xcube dataset for faster access.
      patch     Patch and consolidate the metadata of a xcube dataset CUBE.
      prune     Delete empty chunks.
      resample  Resample data along the time dimension.
      serve     Serve data cubes via web service.
      sh        SentinelHub tools for xcube.
      tile      IMPORTANT NOTE: The xcube tile tool in its current form is...
      vars2dim  Convert cube variables into new dimension.
      verify    Perform cube verification.
      versions  Get versions of important packages used by xcube.

