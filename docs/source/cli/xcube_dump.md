# `xcube dump`

Dump contents of a dataset.

```bash
$ xcube dump --help
```

    
    Usage: xcube dump [OPTIONS] INPUT
    
      Dump contents of an input dataset.
    
    Options:
      -v, --variable, --var <VARIABLE>
                                      Name of a variable (multiple allowed).
      -e, --encoding                  Dump also variable encoding information.
      --help                          Show this message and exit.


Example:

    $ xcube dump xcube_cube.zarr 

