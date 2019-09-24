# `xcube prune`

Delete empty chunks.

Warning: this tool will likely be integrated into ``xcube optimize`` in
the near future.


```bash
$ xcube prune --help
```

    Usage: xcube prune [OPTIONS] CUBE
    
      Delete empty chunks. Deletes all data files associated with empty (NaN-
      only) chunks in given CUBE, which must have ZARR format.
    
    Options:
      --dry-run  Just read and process input, but don't produce any outputs.
      --help     Show this message and exit.


