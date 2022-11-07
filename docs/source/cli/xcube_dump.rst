==============
``xcube dump``
==============

Synopsis
========

Dump contents of a dataset.

::

    $ xcube dump --help

::

    
    Usage: xcube dump [OPTIONS] INPUT

      Dump contents of an input dataset.

    Options:
      --variable, --var VARIABLE  Name of a variable (multiple allowed).
      -E, --encoding              Dump also variable encoding information.
      --help                      Show this message and exit.


Example
=======

::

    $ xcube dump xcube_cube.zarr 

