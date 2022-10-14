================
``xcube verify``
================

Synopsis
========

Perform cube verification.

::

    $ xcube verify --help

::

    Usage: xcube verify [OPTIONS] CUBE

      Perform cube verification.

      The tool verifies that CUBE
      * defines the dimensions "time", "lat", "lon";
      * has corresponding "time", "lat", "lon" coordinate variables and that they
        are valid, e.g. 1-D, non-empty, using correct units;
      * has valid  bounds variables for "time", "lat", "lon" coordinate
        variables, if any;
      * has any data variables and that they are valid, e.g. min. 3-D, all have
        same dimensions, have at least dimensions "time", "lat", "lon".
      * spatial coordinates and their corresponding bounds (if exist) are equidistant
         and monotonically increasing or decreasing.

      If INPUT is a valid xcube dataset, the tool returns exit code 0. Otherwise a
      violation report is written to stdout and the tool returns exit code 3.

    Options:
      --help  Show this message and exit.

Python API
==========

The related Python API functions are

* :py:func:`xcube.core.verify.verify_cube`, and
* :py:func:`xcube.core.verify.assert_cube`.
