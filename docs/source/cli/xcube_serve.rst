.. _demo configuration file: https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config.yml
.. _demo stores configuration file: https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config-with-stores.yml
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

    Usage: xcube serve [OPTIONS] [PATHS...]

      Run the xcube Server for the given configuration and/or the given raster
      dataset paths given by PATHS.

      Each of the PATHS arguments can point to a raster dataset such as a Zarr
      directory (*.zarr), an xcube multi-level Zarr dataset (*.levels), a NetCDF
      file (*.nc), or a GeoTIFF/COG file (*.tiff).

      If one of PATHS is a directory that is not a dataset itself, it is scanned
      for readable raster datasets.

      The --show ASSET option can be used to inspect the current configuration of
      the server. ASSET is one of:

      apis            outputs the list of APIs provided by the server
      endpoints       outputs the list of all endpoints provided by the server
      openapi         outputs the OpenAPI document representing this server
      config          outputs the effective server configuration
      configschema    outputs the JSON Schema for the server configuration

      The ASSET may be suffixed by ".yaml" or ".json" forcing the respective
      output format. The default format is YAML.

      Note, if --show  is provided, the ASSET will be shown and the program will
      exit immediately.

    Options:
      --framework FRAMEWORK           Web server framework. Defaults to "tornado"
      -p, --port PORT                 Service port number. Defaults to 8080
      -a, --address ADDRESS           Service address. Defaults to "0.0.0.0".
      -c, --config CONFIG             Configuration YAML or JSON file.  If
                                      multiple configuration files are passed, they will be merged in
                                      order.
      --base-dir BASE_DIR             Directory used to resolve relative paths in
                                      CONFIG files. Defaults to the parent
                                      directory of (last) CONFIG file.
      --prefix URL_PREFIX             Prefix path to be used for all endpoint
                                      URLs. May include template variables, e.g.,
                                      "api/{version}".
      --revprefix REVERSE_URL_PREFIX  Prefix path to be used for reverse endpoint
                                      URLs that may be reported by server
                                      responses. May include template variables,
                                      e.g., "/proxy/{port}". Defaults to value of
                                      URL_PREFIX.
      --traceperf                     Whether to output extra performance logs.
      --update-after TIME             Check for server configuration updates every
                                      TIME seconds.
      --stop-after TIME               Unconditionally stop service after TIME
                                      seconds.
      --show ASSET                    Show ASSET and exit. Possible values for
                                      ASSET are 'apis', 'endpoints', 'openapi',
                                      'config', 'configschema' optionally suffixed
                                      by '.yaml' or '.json'.
      --open-viewer                   After starting the server, open xcube Viewer
                                      in a browser tab.
      -q, --quiet                     Disable output of log messages to the
                                      console entirely. Note, this will also
                                      suppress error and warning messages.
      -v, --verbose                   Enable output of log messages to the
                                      console. Has no effect if --quiet/-q is
                                      used. May be given multiple times to control
                                      the level of log messages, i.e., -v refers
                                      to level INFO, -vv to DETAIL, -vvv to DEBUG,
                                      -vvvv to TRACE. If omitted, the log level of
                                      the console is WARNING.
      --help                          Show this message and exit.


Configuration File
==================

The xcube server is used to configure the xcube datasets to be published.

xcube datasets are any datasets that

* that comply to `Unidata's CDM <https://docs.unidata.ucar.edu/netcdf-java/current/userguide/common_data_model_overview.html>`_ and to the `CF Conventions <http://cfconventions.org/>`_;
* that can be opened with the `xarray <https://xarray.pydata.org/en/stable/>`_ Python library;
* that have variables that have the dimensions and shape (``lat``, ``lon``) or (``time``, ``lat``, ``lon``);
* that have 1D-coordinate variables corresponding to the dimensions;
* that have their spatial grid defined in arbitrary spatial coordinate reference systems.

The xcube server supports xcube datasets stored as local NetCDF files, as well as
`Zarr <https://zarr.readthedocs.io/en/stable/>`_ directories in the local file system or remote object storage.
Remote Zarr datasets must be stored AWS S3 compatible object storage.

As an example, here is the `configuration of the demo server <https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/config.yml>`_.
The parts of the demo configuration file are explained in detail further down.

Some hints before, which are not addressed in the server demo configuration file.
To increase imaging performance, xcube datasets can be converted to multi-resolution pyramids using the
:doc:`xcube_level` tool. In the configuration, the format must be set to ``'levels'``.
Leveled xcube datasets are configured this way:

.. code-block:: yaml

    Datasets:

      - Identifier: my_multi_level_dataset
        Title: My Multi-Level Dataset
        FileSystem: file
        Path: my_multi_level_dataset.levels

      - ...

To increase time-series extraction performance, xcube datasets may be rechunked with larger chunk size in the ``time``
dimension using the :doc:`xcube_chunk` tool. In the xcube server configuration a hidden dataset is given,
and the it is referred to by the non-hidden, actual dataset using the ``TimeSeriesDataset`` setting:

.. code-block:: yaml

    Datasets:

      - Identifier: my_dataset
        Title: My Dataset
        FileSystem: file
        Path: my_dataset.zarr
        TimeSeriesDataset: my_dataset_opt_for_ts

      - Identifier: my_dataset_opt_for_ts
        Title: My Dataset optimized for Time-Series
        FileSystem: file
        Path: my_ts_opt_dataset.zarr
        Hidden: True

      - ...


.. _config:

Server Demo Configuration File
==============================
The server configuration file consists of various parts, some of them are necessary others are optional.
Here the `demo configuration file`_ used in the `example`_ is explained in detail.

The configuration file consists of five main parts `authentication`_,
`dataset attribution`_, `datasets`_,
`place groups`_ and `styles`_.

.. _authentication:

Authentication [optional]
-------------------------
In order to display data via xcube-viewer exclusively to registered and authorized users, the data served by xcube serve
may be protected by adding Authentication to the server configuration. In order to ensure protection, an *Authority* and an
*Audience* needs to be provided. Here authentication by `Auth0`_ is used.
Please note the trailing slash in the "Authority" URL.

.. code-block:: yaml

    Authentication:
      Authority: https://xcube-dev.eu.auth0.com/
      Audience: https://xcube-dev/api/

Example of OIDC configuration for Keycloak.
Please note that there is no trailing slash in the "Authority" URL.

.. code-block:: yaml

    Authentication:
      Authority: https://kc.brockmann-consult.de/auth/realms/AVL
      Audience: avl-xc-api

.. _dataset attribution:

Dataset Attribution [optional]
------------------------------

Dataset Attribution may be added to the server via *DatasetAttribution*.

.. code-block:: yaml

    DatasetAttribution:
      - "© by Brockmann Consult GmbH 2020, contains modified Copernicus Data 2019, processed by ESA"
      - "© by EU H2020 CyanoAlert project"

.. _base directory:

Base Directory [optional]
-------------------------

A typical xcube server configuration comprises many paths, and
relative paths of known configuration parameters are resolved against
the ``base_dir`` configuration parameter.

.. code-block:: yaml

    base_dir: s3://<bucket>/<path-to-your>/<resources>/

However, for values of
parameters passed to user functions that represent paths in user code,
this cannot be done automatically. For such situations, expressions
can be used. An expression is any string between ``"${"` and `"}"`` in a
configuration value. An expression can contain the variables
``base_dir`` (a string), ``ctx`` the current server context
(type ``xcube.webapi.datasets.DatasetsContext``), as well as the function
``resolve_config_path(path)`` that is used to make a path absolut with
respect to ``base_dir`` and to normalize it. For example

.. code-block:: yaml

    Augmentation:
    Path: augmentation/metadata.py
    Function: metadata:update_metadata
    InputParameters:
        bands_config: ${resolve_config_path("../common/bands.yaml")}


.. _viewer configuration:

Viewer Configuration [optional]
------------------------------

The xcube server endpoint ``/viewer/config/{*path}`` allows
for configuring the viewer accessible via endpoint ``/viewer``.
The actual source for the configuration items is configured by xcube
server configuration using the new entry ``Viewer/Configuration/Path``,
for example:

.. code-block:: yaml

    Viewer:
      Configuration:
        Path: s3://<bucket>/<viewer-config-dir-path>

*Path* [mandatory]
must be an absolute filesystem path or a S3 path as in the example above.
It points to a directory that is expected to contain the the viewer configuration file `config.json` 
among other configuration resources, such as custom ``favicon.ico`` or ``logo.png``.
The file ``config.json`` should conform to the
[configuration JSON Schema](https://github.com/dcs4cop/xcube-viewer/blob/master/src/resources/config.schema.json). 
All its values are optional, if not provided, 
[default values](https://github.com/dcs4cop/xcube-viewer/blob/master/src/resources/config.json) 
are used instead. 

.. _datasets:

Datasets [mandatory]
--------------------

In order to publish selected xcube datasets via ``xcube serve``,
the datasets need to be described in the server configuration.

.. _remotely stored xcube datasets:

Remotely stored xcube Datasets
------------------------------

The following configuration snippet demonstrates how to
publish static (persistent) xcube datasets stored in
S3-compatible object storage:

.. code-block:: yaml

    Datasets:
      - Identifier: remote
        Title: Remote OLCI L2C cube for region SNS
        BoundingBox: [0.0, 50, 5.0, 52.5]
        FileSystem: s3
        Endpoint: "https://s3.eu-central-1.amazonaws.com"
        Path: xcube-examples/OLCI-SNS-RAW-CUBE-2.zarr
        Region: eu-central-1
        Anonymous: true
        Style: default
        ChunkCacheSize: 250M
        PlaceGroups:
          - PlaceGroupRef: inside-cube
          - PlaceGroupRef: outside-cube
        AccessControl:
          RequiredScopes:
            - read:datasets


The above example of how to specify a xcube dataset to be served above is using a datacube stored in
an S3 bucket within the Amazon Cloud. Please have a closer look at the parameter *Anonymous: true*.
This means, the datasets permissions are set to public read in your source s3 bucket. If you have a dataset that is not public-read, set
*Anonymous: false*. Furthermore, you need to have valid credentials on the machine where the server runs.
Credentials may be saved either in a file called .aws/credentials with content like below:

    | [default]
    | aws_access_key_id=AKIAIOSFODNN7EXAMPLE
    | aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

Or they may be exported as environment variables AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID.

Further down an example for a `locally stored xcube datasets`_ will be given,
as well as an example of `dynamic xcube datasets`_.

*Identifier* [mandatory]
is a unique ID for each xcube dataset, it is ment for machine-to-machine interaction
and therefore does not have to be a fancy human-readable name.

*Title* [optional]
should be understandable for humans. This title that will be displayed within the viewer
for the dataset selection. If omitted, the key title from the dataset metadata will be used.
If that is missing too, the identifier will be used.

*BoundingBox* [optional]
may be set in order to restrict the region which is served from a certain datacube. The
notation of the *BoundingBox* is [lon_min,lat_min,lon_max,lat_max].

*FileSystem* [mandatory]
is set to "s3" which lets xcube serve know, that the datacube is located in the cloud.

*Endpoint* [mandatory]
contains information about the cloud provider endpoint, this will differ if you use a different
cloud provider.

*Path* [mandatory]
leads to the specific location of the datacube. The particular datacube is stored in an
OpenTelecomCloud S3 bucket called "xcube-examples" and the datacube is called "OLCI-SNS-RAW-CUBE-2.zarr".

*Region* [optional]
is the region where the specified cloud provider is operating.

*Styles* [optional]
influence the visualization of the xucbe dataset in the xcube viewer if specified in the server configuration file. The usage of *Styles* is described in section `styles`_.

*PlaceGroups* [optional]
allow to associate places (e.g. polygons or point-location) with a particular xcube dataset.
Several different place groups may be connected to a xcube dataset, these different place groups are distinguished by
the *PlaceGroupRef*. The configuration of *PlaceGroups* is described in section `place groups`_.

*AccessControl* [optional]
can only be used when providing `authentication`_. Datasets may be protected by
configuring the *RequiredScopes* entry whose value is a list of required scopes, e.g. "read:datasets".

*Variables* [optional]
enforces the order of variables reported by xcube server.
Is a list of wildcard patterns that
determines the order of variables and the subset of variables to be
reported.

The following example reports only variables whose name starts with "conc\_":

.. code-block:: yaml

  Datasets:
    - Path: demo.zarr
      Variables:
        - "conc_*"

The next example reports all variables but ensures that ``conc_chl``
and ``conc_tsm`` are the first ones:

.. code-block:: yaml

  Datasets:
    - Path: demo.zarr
      Variables:
        - conc_chl
        - conc_tsm
        - "*"


.. _locally stored xcube datasets:

Locally stored xcube Datasets
-----------------------------

The following configuration snippet demonstrates how to
publish static (persistent) xcube datasets stored in the local filesystem:

.. code-block:: yaml

      - Identifier: local
        Title: Local OLCI L2C cube for region SNS
        BoundingBox: [0.0, 50, 5.0, 52.5]
        FileSystem: file
        Path: cube-1-250-250.zarr
        Style: default
        TimeSeriesDataset: local_ts
        Augmentation:
          Path: compute_extra_vars.py
          Function: compute_variables
          InputParameters:
            factor_chl: 0.2
            factor_tsm: 0.7
        PlaceGroups:
          - PlaceGroupRef: inside-cube
          - PlaceGroupRef: outside-cube
        AccessControl:
          IsSubstitute: true

Most of the configuration of locally stored datasets is equal to the configuration of
`remotely stored xcube datasets`_.

*FileSystem* [mandatory]
is set to "file" which lets xcube serve know, that the datacube is locally stored.

*TimeSeriesDataset* [optional]
is not bound to local datasets, this parameter may be used for remotely stored datasets
as well. By using this parameter a time optimized datacube will be used for generating the time series. The configuration
of this time optimized datacube is shown below. By adding *Hidden* with *true* to the dataset configuration, the time optimized
datacube will not appear among the displayed datasets in xcube viewer.

.. code-block:: yaml

  # Will not appear at all, because it is a "hidden" resource
  - Identifier: local_ts
    Title: 'local' optimized for time-series
    BoundingBox: [0.0, 50, 5.0, 52.5]
    FileSystem: file
    Path: cube-5-100-200.zarr
    Hidden: true
    Style: default

*Augmentation* [optional]
augments data cubes by new variables computed on-the-fly, the generation of the on-the-fly
variables depends on the implementation of the python module specified in the *Path* within the *Augmentation*
configuration.

*AccessControl* [optional]
can only be used when providing `authentication`_. By passing the *IsSubstitute* flag
a dataset disappears for authorized requests. This might be useful for showing a demo dataset in the viewer for
user who are not logged in.

.. _dynamic xcube datasets:

Dynamic xcube Datasets
----------------------

There is the possibility to define dynamic xcube datasets
that are computed on-the-fly. Given here is an example that
obtains daily or weekly averages of an xcube dataset named "local".

.. code-block:: yaml

  - Identifier: local_1w
    Title: OLCI weekly L3 cube for region SNS computed from local L2C cube
    BoundingBox: [0.0, 50, 5.0, 52.5]
    FileSystem: memory
    Path: resample_in_time.py
    Function: compute_dataset
    InputDatasets: ["local"]
    InputParameters:
      period: 1W
      incl_stdev: True
    Style: default
    PlaceGroups:
      - PlaceGroupRef: inside-cube
      - PlaceGroupRef: outside-cube
    AccessControl:
      IsSubstitute: True

*FileSystem* [mandatory]
must be "memory" for dynamically generated datasets.

*Path* [mandatory]
points to a Python module. Can be a Python file, a package, or a Zip file.

*Function* [mandatory, mutually exclusive with *Class*]
references a function in the Python file given by *Path*. Must be suffixed
by colon-separated module name, if *Path* references a package or Zip file.
The function receives one or more datasets of type ``xarray.Dataset``
as defined by *InputDatasets* and optional keyword-arguments as
given by *InputParameters*, if any. It must return a new ``xarray.Dataset``
with same spatial coordinates as the inputs.
If "resample_in_time.py" is compressed among any other modules in a zip archive, the original module name
must be indicated by the prefix to the function name:

.. code-block:: yaml

    Path: modules.zip
    Function: resample_in_time:compute_dataset
    InputDatasets: ["local"]


*Class* [mandatory, mutually exclusive with *Function*]
references a callable in the Python file given by *Path*. Must be suffixed
by colon-separated module name, if *Path* references a package or Zip file.
The callable is either a class derived from
``xcube.core.mldataset.MultiLevelDataset`` or a function that returns
an instance of ``xcube.core.mldataset.MultiLevelDataset``.
The callable receives one or more datasets of type
``xcube.core.mldataset.MultiLevelDataset`` as defined by *InputDatasets*
and optional keyword-arguments as given by *InputParameters*, if any.

*InputDatasets* [mandatory]
specifies the input datasets passed to *Function* or *Class*.

*InputParameters* [mandatory]
specifies optional keyword arguments passed to *Function* or *Class*.
In the example, *InputParameters* defines which kind of resampling
should be performed.

Again, the dataset may be associated with place groups.

.. _place groups:

Place Groups [optional]
-----------------------

Place groups are specified in a similar manner compared to specifying datasets within a server.
Place groups may be stored e.g. in shapefiles or a geoJson.

.. code-block:: yaml

    PlaceGroups:
      - Identifier: outside-cube
        Title: Points outside the cube
        Path: places/outside-cube.geojson
        PropertyMapping:
          image: ${resolve_config_path("images/outside-cube/${ID}.jpg")}


*Identifier* [mandatory]
is a unique ID for each place group, it is the one xcube serve uses to associate
a place group to a particular dataset.

*Title* [mandatory]
should be understandable for humans and this is the title that will be displayed within the viewer
for the place selection if the selected xcube dataset contains a place group.

*Path* [mandatory]
defines where the file storing the place group is located.
Please note that the paths within the example config are relative.

*PropertyMapping* [mandatory]
determines which information contained within the place group should be used for selecting a certain location of the given place group.
This depends very strongly of the data used. In the above example, the image URL is determined by a feature's ``ID`` property.

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

.. code-block:: yaml

    PlaceGroups:
        Identifier: my_group
        ...
        PropertyMapping:
            label: NAME
            image: IMG_URL


The values on the right side may either **be** feature property names or **contain** them as placeholders in the form
``${PROPERTY}``.

.. _styles:

Styles [optional]
-----------------


Within the *Styles* section, colorbars may be defined which should
be used initially for a certain variable of a dataset,
as well as the value ranges. For xcube viewer version 0.3.0 or
higher the colorbars and the value ranges may be adjusted by the user
within the xcube viewer.

.. code-block:: yaml

    Styles:
      - Identifier: default
        ColorMappings:
          conc_chl:
            ColorBar: plasma
            ValueRange: [0., 24.]
          conc_tsm:
            ColorBar: PuBuGn
            ValueRange: [0., 100.]
          kd489:
            ColorBar: jet
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

The *ColorMapping* may be specified for each variable of the
datasets to be served. If not specified, xcube server will try
to extract default values from attributes of dataset variables.
The default value ranges are determined by:

* xcube-specific variable attributes
  ``color_value_min`` and ``color_value_max``;
* The CF variable attributes ``valid_min``, ``valid_max``
  or ``valid_range``.
* Or otherwise, the value range ``[0, 1]`` is assumed.

The colorbar name can be set using the

* xcube-specific variable attribute ``color_bar_name``;
* Otherwise, the default colorbar name will be ``viridis``.

The special name *rgb* may be used to generate an RGB-image
from any other three dataset variables used for the individual
*Red*, *Green* and *Blue* channels of the resulting image.
An example is shown in the configuration above.

Colormaps may be reversed by using name suffix "_r".
They also can have alpha blending indicated by name suffix "_alpha".
Both, reversed and alpha blending is possible as well and can be configured by name suffix "_r_alpha".

.. code-block:: yaml

    Styles:
      - Identifier: default
        ColorMappings:
          conc_chl:
            ColorBar: plasma_r_alpha
            ValueRange: [0., 24.]

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


Server Demo Configuration File for DataStores
=============================================
The server configuration file consists of various parts, some of them are necessary, others are optional.
Here the `demo stores configuration file`_ used in the `example stores`_ is explained in detail.

This configuration file differs only in one part compared to section :ref:`Server Demo Configuration File <config>`:
`data stores`_.
The other main parts (`authentication`_, `dataset attribution`_,
`place groups`_, and `styles`_) can be used in combination with `data stores`_.

.. _data stores:

DataStores [mandatory]
--------------------

Datasets, which are stored in the same location, may be configured in the configuration file using *DataStores*.


.. code-block:: yaml

    DataStores:
      - Identifier: edc
        StoreId: s3
        StoreParams:
          root: xcube-dcfs/edc-xc-viewer-data
          max_depth: 1
          storage_options:
            anon: true
            # client_kwargs:
            #  endpoint_url: https://s3.eu-central-1.amazonaws.com
        Datasets:
          - Path: "*2.zarr"
            Style: default
            # ChunkCacheSize: 1G


*Identifier* [mandatory]
is a unique ID for each DataStore.

*StoreID* [mandatory]
can be *file* for locally stored datasets and *s3* for datasets located in the cloud.

| *StoreParams* [mandatory]
| *root* [mandatory] states a common beginning part of the paths of the served datasets.
| *max_depth* [optional] if wildcard is used in *Dataset Path* this indicated how far the server should step down and serve the discovered datasets.
| *storage_options* [optional] is necessary when serving datasets from the cloud. With *anon* the accessibility is configured, if the datasets are public-read, *anon* is set to "true", "false" indicates they are protected. Credentials may be set by keywords *key* and *secret*.

*Datasets* [optional]
if not specified, every dataset in the indicated location supported by xcube will be read and
served by xcube serve. In order to filter certain datasets you can list Paths that shall be served by xcube serve.
*Path* may contain wildcards. Each Dataset entry may have *Styles* and *PlaceGroups* associated with them, the same way
as described in section :ref:`Server Demo Configuration File <config>`.

.. _example stores:

Example Stores
==============

::

    xcube serve --port 8080 --config ./examples/serve/demo/config-with-stores.yml --verbose

::

    xcube Server: WMTS, catalogue, data access, tile, feature, time-series services for xarray-enabled data cubes, version
    [I 190924 17:08:54 service:228] configuration file 'D:\\Projects\\xcube\\examples\\serve\\demo\\config.yml' successfully loaded
    [I 190924 17:08:54 service:158] service running, listening on localhost:8080, try http://localhost:8080/datasets
    [I 190924 17:08:54 service:159] press CTRL+C to stop service

.. _example azure blob storage filesystem stores:

Example Azure Blob Storage filesystem Stores
============================================

xcube server includes support for Azure Blob Storage filesystem by a data store `abfs`.
This enables access to data cubes (`.zarr` or `.levels`) in Azure blob storage as shown here:


.. code-block:: yaml

    DataStores:
      - Identifier: siec
        StoreId: abfs
        StoreParams:
          root: my_blob_container
          max_depth: 1
          storage_options:
            anon: true
            account_name: "xxx"
            account_key': "xxx"
            # or
            # connection_string: "xxx"
        Datasets:
          - Path: "*.levels"
            Style: default


Web API
=======

The xcube server has a dedicated self describing Web API Documentation. After starting the server, you can check the
various functions provided by xcube Web API. To explore the functions, open ``<base-url>/openapi.html``.

The xcube server implements the OGC WMTS RESTful and KVP architectural styles of the
`OGC WMTS 1.0.0 specification <http://www.opengeospatial.org/standards/wmts>`_. The following operations are supported:

* **GetCapabilities**: ``/xcube/wmts/1.0.0/WMTSCapabilities.xml``
* **GetTile**: ``/xcube/wmts/1.0.0/tile/{DatasetName}/{VarName}/{TileMatrix}/{TileCol}/{TileRow}.png``
* **GetFeatureInfo**: *in progress*

