# `xcube dump`

Dump contents of a dataset.

```bash
    $ xcube dump --help
```

    
    Usage: xcube dump [OPTIONS] <path>
    
    Dump contents of a dataset.
    
    optional arguments:
      --help                Show this help message and exit
      --variable, -v        Name of a variable (multiple allowed).
      --encoding, -e        Dump also variable encoding information.


Example:

    $ xcube dump xcube_cube.zarr 

