=================
``xcube extract``
=================

Synopsis
========

Extract cube points.

::

    $ xcube extract --help

::

    Usage: xcube extract [OPTIONS] CUBE POINTS

      Extract cube points.

      Extracts data cells from CUBE at coordinates given in each POINTS record and
      writes the resulting values to given output path and format.

      POINTS must be a CSV file that provides at least the columns "lon", "lat",
      and "time". The "lon" and "lat" columns provide a point's location in
      decimal degrees. The "time" column provides a point's date or date-time. Its
      format should preferably be ISO, but other formats may work as well.

    Options:
      -o, --output OUTPUT  Output path. If omitted, output is written to stdout.
      -f, --format FORMAT  Output format. Currently, only 'csv' is supported.
      -C, --coords         Include cube cell coordinates in output.
      -B, --bounds         Include cube cell coordinate boundaries (if any) in
                           output.
      -I, --indexes        Include cube cell indexes in output.
      -R, --refs           Include point values as reference in output.
      --help               Show this message and exit.


Example
=======

::

    $ xcube extract xcube_cube.zarr -o point_data.csv -Cb --indexes --refs


Python API
==========

Related Python API functions are

* :py:func:`xcube.core.extract.get_cube_values_for_points`,
* :py:func:`xcube.core.extract.get_cube_point_indexes`, and
* :py:func:`xcube.core.extract.get_cube_values_for_indexes`.
