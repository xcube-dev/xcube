===============
``xcube prune``
===============

Delete empty chunks.

.. attention:: This tool will likely be integrated into ``xcube optimize`` in the near future.


::

    $ xcube prune --help

::

    Usage: xcube prune [OPTIONS] DATASET

      Delete empty chunks. Deletes all data files associated with empty (NaN-only)
      chunks in given DATASET, which must have Zarr format.

    Options:
      -q, --quiet    Disable output of log messages to the console entirely. Note,
                     this will also suppress error and warning messages.
      -v, --verbose  Enable output of log messages to the console. Has no effect
                     if --quiet/-q is used. May be given multiple times to control
                     the level of log messages, i.e., -v refers to level INFO, -vv
                     to DETAIL, -vvv to DEBUG, -vvvv to TRACE. If omitted, the log
                     level of the console is WARNING.
      --dry-run      Just read and process input, but don't produce any output.
      --help         Show this message and exit.


A related Python API function is :py:func:`xcube.core.optimize.get_empty_dataset_chunks`.
