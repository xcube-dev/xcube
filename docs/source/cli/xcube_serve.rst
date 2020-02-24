.. _demo configuration file: https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config.yml
.. _Auth0: https://auth0.com/

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
The parts of the demo configuration file are explained in detail further down.

Some hints before, which are not addressed in the server demo configuration file.
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


Server Demo Configuration File
==============================
The server configuration file consists of various parts, some of them are necessary others are optional.
Here the `demo configuration file`_ used in the :ref:`example` is explained in detail.

The configuration file consists of five main parts :ref:`authentication`, :ref:`dataset-attribution`, :ref:`datasets`,
:ref:`place-groups` and :ref:`styles`.

.. _authentication:
Authentication [optional]
-------------------------
In order to display data via xcube-viewer exclusively to registered and authorized users, the data served by xcube serve
may be protected by adding Authentication to the server configuration. In order to ensure protection, a *Domain* and an
*Audience* needs to be provided. Here authentication by `Auth0`_ is used.

.. code:: yaml

    Authentication:
      Domain: xcube-dev.eu.auth0.com
      Audience: https://xcube-dev/api/

.. _dataset-attribution:
Dataset Attribution [optional]
------------------------------

Dataset Attribution may be added to the server via *DatasetAttribution*.

.. code:: yaml

    DatasetAttribution:
      - "© by Brockmann Consult GmbH 2020, contains modified Copernicus Data 2019, processed by ESA"
      - "© by EU H2020 CyanoAlert project"


.. _datasets:
Datasets [mandatory]
--------------------
In order to publish selected xcube datasets via xcube serve the datasets need to be specified in the configuration
file of the server. Several xcube datasets may be served within one server, by providing a list of information
concerning the xcube datasets.

.. _remotely-stored-xcube-datasets:
Remotely Stored xcube Datasets
-----------------------------
.. code:: yaml

    Datasets:
      - Identifier: remote
        Title: Remote OLCI L2C cube for region SNS
        BoundingBox: [0.0, 50, 5.0, 52.5]
        FileSystem: obs
        Endpoint: "http://obs.eu-de.otc.t-systems.com"
        Path: "dcs4cop-obs-02/OLCI-SNS-RAW-CUBE-2.zarr"
        Region: "eu-de"
        Style: default
        PlaceGroups:
          - PlaceGroupRef: inside-cube
          - PlaceGroupRef: outside-cube
        AccessControl:
          RequiredScopes:
            - read:datasets


The above example of how to specify a xcube dataset to be served above is using a datacube stored in
an S3 bucket within the OpenTelekomCloud.
Further down an example for a :ref:`locally-stored-xcube-datasets` will be given,
as well as an example of a :ref:`on-the-fly-generation-of-xcube-datasets`.

*Identifier* [mandatory] is a unique ID for each xcube dataset, it is ment for machine-to-machine interaction
and therefore does not have to be a fancy human-readable name.

*Title* [mandatory] should be understandable for humans and this is the title that will be displayed within the viewer
for the dataset selection.

*BoundingBox* [optional] may be set in order to restrict the region which is served from a certain datacube. The
notation of the *BoundingBox* is [lon_min,lat_min,lon_max,lat_max].

*FileSystem* [mandatory] is set to "obs" which lets xcube serve know, that the datacube is located in the cloud
(*obs* is the abbreviation for object storage).

*Endpoint* [mandatory] contains information about the cloud provider endpoint, this will differ if you use a different
cloud provider.

*Path* [mandatory] leads to the specific location of the datacube. The particular datacube is stored in an
OpenTelecomCloud S3 bucket called "dcs4cop-obs-02" and the datacube is called "OLCI-SNS-RAW-CUBE-2.zarr".

*Region* [mandatory] is the region where the specified cloud provider is operating.

*Styles* [optional] influence the visualization of the xucbe dataset in the xcube viewer if specified
in the server configuration file. The usage of *Styles* is described in section :ref:`styles`.

*PlaceGroups* [optional] allow to associate places (e.g. polygons or point-location) with a particular xcube dataset.
Several different place groups may be connected to a xcube dataset, these different place groups are distinguished by
the *PlaceGroupRef*. The configuration of *PlaceGroups* is described in section :ref:`place-groups`.

*AccessControl* [optional] can only be used when providing :ref:`authentication`. Datasets may be protected by
configuring the *RequiredScopes* entry whose value is a list of required scopes, e.g. "read:datasets".

.. _locallly-stored-xcube-datasets:
Locally Stored xcube Datasets
-----------------------------

To serve a locally stored dataset the configuration of it would look like the example below:

.. code:: yaml

      - Identifier: local
        Title: "Local OLCI L2C cube for region SNS"
        BoundingBox: [0.0, 50, 5.0, 52.5]
        FileSystem: local
        Path: cube-1-250-250.zarr
        Style: default
        TimeSeriesDataset: local_ts
        Augmentation:
          Path: "compute_extra_vars.py"
          Function: "compute_variables"
          InputParameters:
            factor_chl: 0.2
            factor_tsm: 0.7
        PlaceGroups:
          - PlaceGroupRef: inside-cube
          - PlaceGroupRef: outside-cube
        AccessControl:
          IsSubstitute: true

Most of the configuration of locally stored datasets is equal to the configuration of
:ref:`remotely-stored-xcube-datasets`.

*FileSystem* [mandatory] is set to "local" which lets xcube serve know, that the datacube is locally stored.

*TimeSeriesDataset* [optional] is not bound to local datasets, this parameter may be used for remotely stored datasets
as well. By using this parameter a time optimized datacube will be used for generating the time series. The configuration
of this time optimized datacube is shown below. By adding *Hidden* with *true* to the dataset configuration, the time optimized
datacube will not appear among the displayed datasets in xucbe viewer.

.. code:: yaml

  # Will not appear at all, because it is a "hidden" resource
  - Identifier: local_ts
    Title: "'local' optimized for time-series"
    BoundingBox: [0.0, 50, 5.0, 52.5]
    FileSystem: local
    Path: cube-5-100-200.zarr
    Hidden: true
    Style: default

*Augmentation* [optional] augments data cubes by new variables computed on-the-fly, the generation of the on-the-fly
variables depends on the implementation of the python module specified in the *Path* within the *Augmentation*
configuration.

*AccessControl* [optional] can only be used when providing :ref:`authentication`. By passing the *IsSubstitute* flag
a dataset disappears for authorized requests. This might be useful for showing a demo dataset in the viewer for
user who are not logged in.

.. _on-the-fly-generation-of-xcube-datasets:
On-the-fly Generation of xcube Datasets
---------------------------------------

There is the possibility of generating resampled xcube datasets on-the-fly, e.g. in order to
obtain daily or weekly averages of a xcube dataset.

.. code:: yaml

  - Identifier: local_1w
    Title: OLCI weekly L3 cube for region SNS computed from local L2C cube
    BoundingBox: [0.0, 50, 5.0, 52.5]
    FileSystem: memory
    Path: "resample_in_time.py"
    Function: "compute_dataset"
    InputDatasets: ["local"]
    InputParameters:
      period: "1W"
      incl_stdev: True
    Style: default
    PlaceGroups:
      - PlaceGroupRef: inside-cube
      - PlaceGroupRef: outside-cube
    AccessControl:
      IsSubstitute: True

*FileSystem* [mandatory] is defined as "memory" for the on-the-fly generated dataset.

*Path* [mandatory] leads to the resample python module. There might be several functions specified in the
python module, therefore the particular *Function* needs to be included into the configuration.

*InputDatasets* [mandatory] specifies the dataset to be resampled.

*InputParameter* [mandatory] defines which kind of resampling should be performed.
In the example a weekly average is computed.

Again, the dataset may be associated with place groups.

.. _place-groups:
Place Groups [optional]
-----------------------

Place groups are specified in a similar manner compared to specifying datasets within a server.
Place groups may be stored e.g. in shapefiles or a geoJson.

.. code:: yaml

    PlaceGroups:
      - Identifier: outside-cube
        Title: Points outside the cube
        Path: "places/outside-cube.geojson"
        PropertyMapping:
          image: "${base_url}/images/outside-cube/${ID}.jpg"


*Identifier* [mandatory] is a unique ID for each place group, it is the one xcube serve uses to associate
a place group to a particular dataset.

*Title* [mandatory] should be understandable for humans and this is the title that will be displayed within the viewer
for the place selection if the selected xcube dataset contains a place group.

*Path* [mandatory] defines where the file storing the place group is located.
Please note that the paths within the example config are relative.

*PropertyMapping* [mandatory] determines which information contained within the place group
should be used for selecting a certain location of the given place group. This depends very
strongly of the data used. In the above example, the image URL is determined by a feature's ``ID`` property.

Property Mappings
-----------------

The entry *PropertyMapping* is used to map a set of well-known properties (or roles) to the actual properties provided
by a place feature in a place group. For example, the well-known properties are used to in xcube viewer to display
information about the currently selected place.
The possible well-known properties are:

* ``label``: The property that provides a label for the place, if any.
  Defaults to to case-insensitive names ``label``, ``title``, ``name``, ``id`` in xcube viewer.
* ``color``: The property that provides a place's color.
  Defaults to the case-insensitive name ``color`` in xcube viewer.
* ``image``: The property that provides a place's image URL, if any.
  Defaults to case-insensitive names ``image``, ``img``, ``picture``, ``pic`` in xcube viewer.
* ``description``: The property that provides a place's description text, if any.
  Defaults to case-insensitive names ``description``, ``desc``, ``abstract``, ``comment`` in xcube viewer.


In the following example, a place's label is provided by the place feature's ``NAME`` property,
while an image is provided by the place feature's ``IMG_URL`` property:

.. code:: yaml

    PlaceGroups:
        Identifier: my_group
        ...
        PropertyMapping:
            label: NAME
            image: IMG_URL


The values on the right side may either **be** feature property names or **contain** them as placeholders in the form
``${PROPERTY}``. A special placeholder is ``${base_url}`` which is replaced by the server's current base URL.

.. _styles:
Styles [optional]
-----------------


Within the *Styles* section colorbars may be defined which should be used initially for a certain variable of a dataset,
as well as the value ranges.
For xcube viewer version 0.3.0 or higher the colorbars and the value ranges may be adjusted by the user
within the xcube viewer.

.. code:: yaml

    Styles:
      - Identifier: default
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
          rgb:
            Red:
              Variable: conc_chl
              ValueRange: [0., 24.]
            Green:
              Variable: conc_tsm
              ValueRange: [0., 100.]
            Blue:
              Variable: kd489
              ValueRange: [0., 6.]

The *ColorMapping* may be specified for each variable of the datasets to be served.
If not specified, the server uses a default colorbar as well as a default value range.

*rgb* may be used to generate an RGB-Image on-the-fly within xcube viewer. This may be done if  the dataset contains
variables which represent the bands red, green and blue, they may be combined to an RGB-Image. Or three variables
of the dataset may be combined to an RGB-Image, as shown in the configuration above.

.. _example:
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


