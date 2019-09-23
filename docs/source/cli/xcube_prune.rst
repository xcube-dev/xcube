===============
``xcube prune``
===============

Delete empty chunks.

Warning: this tool will likely be integrated into ``xcube optimize`` in
the near future.


::

    $ xcube prune --help

::

    Usage: xcube prune [OPTIONS] CUBE
    
      Delete empty chunks. Deletes all data files associated with empty (NaN-
      only) chunks in given CUBE, which must have ZARR format.
    
    Options:
      --dry-run  Just read and process input, but don't produce any outputs.
      --help     Show this message and exit.


A related Python API function is :py:func:`xcube.api.get_empty_dataset_chunks`.
