.. _`TMS 1.0 Specification`: https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification
.. _`YAML format`: https://en.wikipedia.org/wiki/YAML

==============
``xcube tile``
==============

IMPORTANT NOTE:
    The xcube tile tool in its current form is deprecated and no
    longer supported since xcube 0.11. A new tool is planned that can work
    concurrently on dask clusters and also supports common tile grids such as
    global geographic and web mercator.

Synopsis
========

Generate a tiled RGB image pyramid from any xcube dataset.

The format and file organisation of the generated tile sets conforms to the `TMS 1.0 Specification`_.

An optional configuration file given by the `-c` option uses `YAML format`_.

::

    $ xcube tile --help

::

    Usage: xcube tile [OPTIONS] CUBE

      IMPORTANT NOTE: The xcube tile tool in its current form is deprecated and no
      longer supported since xcube 0.11. A new tool is planned that can work
      concurrently on dask clusters and also supports common tile grids such as
      global geographic and web mercator.

      Create RGBA tiles from CUBE.

      Color bars and value ranges for variables can be specified in a CONFIG file.
      Here the color mappings are defined for a style named "ocean_color":

      Styles:
        - Identifier: ocean_color
          ColorMappings:
            conc_chl:
              ColorBar: "plasma"
              ValueRange: [0., 24.]
            conc_tsm:
              ColorBar: "PuBuGn"
              ValueRange: [0., 100.]
            kd489:
              ColorBar: "jet"
              ValueRange: [0., 6.]

      This is the same styles syntax as the configuration file for "xcube serve",
      hence its configuration can be reused.

    Options:
      --variables, --vars VARIABLES  Variables to be included in output. Comma-
                                     separated list of names which may contain
                                     wildcard characters "*" and "?".
      --labels LABELS                Labels for non-spatial dimensions, e.g.
                                     "time=2019-20-03". Multiple values are
                                     separated by comma.
      -t, --tile-size TILE_SIZE      Tile size in pixels for individual or both x-
                                     and y-directions. Separate by comma for
                                     individual tile sizes, e.g. "-t 360,180".
                                     Defaults to the chunks sizes in x- and
                                     y-directions of CUBE, which may not be ideal.
                                     Use option --dry-run and --verbose to display
                                     the default tile sizes for CUBE.
      -c, --config CONFIG            Configuration file in YAML format.
      -s, --style STYLE              Name of a style identifier in CONFIG file.
                                     Only used if CONFIG is given. Defaults to
                                     'default'.
      -o, --output OUTPUT            Output path. Defaults to 'out.tiles'
      -q, --quiet                    Disable output of log messages to the console
                                     entirely. Note, this will also suppress error
                                     and warning messages.
      -v, --verbose                  Enable output of log messages to the console.
                                     Has no effect if --quiet/-q is used. May be
                                     given multiple times to control the level of
                                     log messages, i.e., -v refers to level INFO,
                                     -vv to DETAIL, -vvv to DEBUG, -vvvv to TRACE.
                                     If omitted, the log level of the console is
                                     WARNING.
      --dry-run                      Generate all tiles but don't write any files.
      --help                         Show this message and exit.





Example
=======

An example that uses a configuration file only::

```bash
    xcube tile https://s3.eu-central-1.amazonaws.com/esdl-esdc-v2.0.0/esdc-8d-0.083deg-1x2160x4320-2.0.0.zarr \
      --labels time='2009-01-01/2009-12-30' \
      --vars analysed_sst,air_temperature_2m \
      --tile-size 270 \
      --config ./config-cci-cfs.yml \
      --style default \
      --verbose
```

The configuration file `config-cci-cfs.yml` content is:

```yaml
    Styles:
      - Identifier: default
        ColorMappings:
          analysed_sst:
            ColorBar: "inferno"
            ValueRange: [270, 310]
          air_temperature_2m:
            ColorBar: "magma"
            ValueRange: [190, 320]
```

Python API
==========

There is currently no related Python API.


