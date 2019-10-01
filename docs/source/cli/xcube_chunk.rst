===============
``xcube chunk``
===============

Synopsis
========

(Re-)chunk xcube dataset.

::

    $ xcube chunk --help

::
    
    Usage: xcube chunk [OPTIONS] CUBE

      (Re-)chunk xcube dataset. Changes the external chunking of all variables
      of CUBE according to CHUNKS and writes the result to OUTPUT.

    Options:
      -o, --output OUTPUT  Output path. Defaults to 'out.zarr'
      -f, --format FORMAT  Format of the output. If not given, guessed from
                           OUTPUT.
      -p, --params PARAMS  Parameters specific for the output format. Comma-
                           separated list of <key>=<value> pairs.
      -C, --chunks CHUNKS  Chunk sizes for each dimension. Comma-separated list of
                           <dim>=<size> pairs, e.g. "time=1,lat=270,lon=270"
      --help               Show this message and exit.


Example
=======

::

    $ xcube chunk input_not_chunked.zarr -o output_rechunked.zarr --chunks "time=1,lat=270,lon=270"

Python API
==========

The related Python API function is :py:func:`xcube.api.chunk_dataset`.