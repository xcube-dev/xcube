===============
``xcube serve``
===============

Synopsis
========

Serve data cubes via web service.

``xcube serve`` starts a light-weight web server that provides various services based on xcube datasets:

* Catalogue services to query for xcube datasets and their variables and dimensions, and feature collections;
* Tile map service, with some OGC WMTS 1.0 compatibility (REST and KVP APIs);
* Dataset services to extract subsets like time-series and profiles for e.g. JavaScript clients.

::

    $ xcube serve --help

::

    Usage: xcube serve [OPTIONS] [CUBE]...

      Serve data cubes via web service.

      Serves data cubes by a RESTful API and a OGC WMTS 1.0 RESTful and KVP
      interface. The RESTful API documentation can be found at
      https://app.swaggerhub.com/apis/bcdev/xcube-server.

    Options:
      -A, --address ADDRESS  Service address. Defaults to 'localhost'.
      -P, --port PORT        Port number where the service will listen on.
                             Defaults to 8080.
      --prefix PREFIX        Service URL prefix. May contain template patterns
                             such as "${version}" or "${name}". For example
                             "${name}/api/${version}".
      -u, --update PERIOD    Service will update after given seconds of
                             inactivity. Zero or a negative value will disable
                             update checks. Defaults to 2.0.
      -S, --styles STYLES    Color mapping styles for variables. Used only, if one
                             or more CUBE arguments are provided and CONFIG is not
                             given. Comma-separated list with elements of the form
                             <var>=(<vmin>,<vmax>) or
                             <var>=(<vmin>,<vmax>,"<cmap>")
      -c, --config CONFIG    Use datasets configuration file CONFIG. Cannot be
                             used if CUBES are provided.
      --tilecache SIZE       In-memory tile cache size in bytes. Unit suffixes
                             'K', 'M', 'G' may be used. Defaults to '512M'. The
                             special value 'OFF' disables tile caching.
      --tilemode MODE        Tile computation mode. This is an internal option
                             used to switch between different tile computation
                             implementations. Defaults to 0.
      -s, --show             Run viewer app. Requires setting the environment
                             variable XCUBE_VIEWER_PATH to a valid xcube-viewer
                             deployment or build directory. Refer to
                             https://github.com/dcs4cop/xcube-viewer for more
                             information.
      -v, --verbose          Delegate logging to the console (stderr).
      --traceperf            Print performance diagnostics (stdout).
      --help                 Show this message and exit.


Configuration File
==================

The xcube server is used to configure the xcube datasets to be published.

xcube datasets are any datasets that

* that comply to `Unidata's CDM <https://www.unidata.ucar.edu/software/thredds/v4.3/netcdf-java/CDM/>`_ and to the `CF Conventions <http://cfconventions.org/>`_;
* that can be opened with the `xarray <https://xarray.pydata.org/en/stable/>`_ Python library;
* that have variables that have at least the dimensions and shape (``time``, ``lat``, ``lon``), in exactly this order;
* that have 1D-coordinate variables corresponding to the dimensions;
* that have their spatial grid defined in the WGS84 (``EPSG:4326``) coordinate reference system.

The xcube server supports xcube datasets stored as local NetCDF files, as well as
`Zarr <https://zarr.readthedocs.io/en/stable/>`_ directories in the local file system or remote object storage.
Remote Zarr datasets must be stored in publicly accessible, AWS S3 compatible object storage (OBS).

As an example, here is the `configuration of the demo server <https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config.yml>`_.

To increase imaging performance, xcube datasets can be converted to multi-resolution pyramids using the
:doc:`cli/xcube_level` tool. In the configuration, the format must be set to ``'level'``.
Leveled xcube datasets are configured this way:

.. code:: yaml

    Datasets:

      - Identifier: my_multi_level_dataset
        Title: "My Multi-Level Dataset"
        FileSystem: local
        Path: my_multi_level_dataset.level
        Format: level

      - ...

To increase time-series extraction performance, xcube datasets my be rechunked with larger chunk size in the ``time``
dimension using the :doc:`cli/xcube_chunk` tool. In the xcube server configuration a hidden dataset is given,
and the it is referred to by the non-hidden, actual dataset using the ``TimeSeriesDataset`` setting:

.. code:: yaml

    Datasets:

      - Identifier: my_dataset
        Title: "My Dataset"
        FileSystem: local
        Path: my_dataset.zarr
        TimeSeriesDataset: my_dataset_opt_for_ts

      - Identifier: my_dataset_opt_for_ts
        Title: "My Dataset optimized for Time-Series"
        FileSystem: local
        Path: my_ts_opt_dataset.zarr
        Format: zarr
        Hidden: True

      - ...


Example
=======

::

    xcube serve --port 8080 --config ./examples/serve/demo/config.yml --verbose

::

    xcube Server: WMTS, catalogue, data access, tile, feature, time-series services for xarray-enabled data cubes, version 0.2.0
    [I 190924 17:08:54 service:228] configuration file 'D:\\Projects\\xcube\\examples\\serve\\demo\\config.yml' successfully loaded
    [I 190924 17:08:54 service:158] service running, listening on localhost:8080, try http://localhost:8080/datasets
    [I 190924 17:08:54 service:159] press CTRL+C to stop service


Web API
=======

The xcube server has a dedicated `Web API Documentation <https://app.swaggerhub.com/apis-docs/bcdev/xcube-server>`_
on SwaggerHub. It also lets you explore the API of existing xcube-servers.

The xcube server implements the OGC WMTS RESTful and KVP architectural styles of the
`OGC WMTS 1.0.0 specification <http://www.opengeospatial.org/standards/wmts>`_. The following operations are supported:

* **GetCapabilities**: ``/xcube/wmts/1.0.0/WMTSCapabilities.xml``
* **GetTile**: ``/xcube/wmts/1.0.0/tile/{DatasetName}/{VarName}/{TileMatrix}/{TileCol}/{TileRow}.png``
* **GetFeatureInfo**: *in progress*


