==================
``xcube vars2dim``
==================

Synopsis
========

Convert cube variables into new dimension.

::

    $ xcube vars2dim --help

::
    
    Usage: xcube vars2dim [OPTIONS] CUBE

      Convert cube variables into new dimension. Moves all variables of CUBE
      into a single new variable <var-name> with a new dimension DIM-NAME and
      writes the results to OUTPUT.

    Options:
      --variable, --var VARIABLE  Name of the new variable that includes all
                                  variables. Defaults to "data".
      -D, --dim_name DIM-NAME     Name of the new dimension into variables.
                                  Defaults to "var".
      -o, --output OUTPUT         Output path. If omitted, 'INPUT-vars2dim.FORMAT'
                                  will be used.
      -f, --format FORMAT         Format of the output. If not given, guessed from
                                  OUTPUT.
      --help                      Show this message and exit.


Python API
==========

The related Python API function is :py:func:`xcube.core.vars2dim.vars_to_dim`.
