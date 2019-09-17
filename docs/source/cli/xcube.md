# `xcube`

```bash
$ xcube --help
```

    Usage: xcube [OPTIONS] COMMAND [ARGS]...
    
    Xcube Toolkit
    
    Options:
      --version                Show the version and exit.
      --traceback              Enable tracing back errors by dumping the Python
                               call stack. Pass as very first option to also trace
                               back error during command-line validation.
      --scheduler <scheduler>  Enable distributed computing using the Dask
                               scheduler identified by <scheduler>. <scheduler>
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
      chunk     (Re-)chunk data cube.
      dump      Dump contents of an input dataset.
      extract   Extract cube points.
      gen       Generate data cube.
      grid      Find spatial data cube resolutions and adjust bounding boxes.
      level     Generate multi-resolution levels.
      optimize  Optimize data cube for faster access.
      prune     Delete empty chunks.
      resample  Resample data along the time dimension.
      serve     Serve data cubes via web service.
      vars2dim  Convert cube variables into new dimension.
      verify    Perform cube verification.
