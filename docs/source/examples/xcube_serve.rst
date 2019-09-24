.. warning:: This chapter is a work in progress and currently less than a draft.

=========================
Publishing xcube datasets
=========================

This example demonstrates how to run an xcube server to publish existing xcube datasets.

Running the server
==================

To run the server on default port 8080 using the demo configuration:

::

    $ xcube serve --verbose -c examples/serve/demo/config.yml

To run the server using a particular xcube dataset path and styling information for a variable:

::

    $ xcube serve --styles conc_chl=(0,20,"viridis") examples/serve/demo/cube-1-250-250.zarr


Test it
=======

After starting the server, check the various functions provided by xcube Web API.

* Datasets:
    * `Get datasets <http://localhost:8080/datasets>`_
    * `Get dataset details <http://localhost:8080/datasets/local>`_
    * `Get dataset coordinates <http://localhost:8080/datasets/local/coords/time>`_
* Color bars:
    * `Get color bars <http://localhost:8080/colorbars>`_
    * `Get color bars (HTML) <http://localhost:8080/colorbars.html>`_
* WMTS:
    * `Get WMTS KVP Capabilities (XML) <http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetCapabilities>`_
    * `Get WMTS KVP local tile (PNG) <http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetTile&Version=1.0.0&Layer=local.conc_chl&TileMatrix=0&TileRow=0&TileCol=0&Format=image/png>`_
    * `Get WMTS KVP remote tile (PNG) <http://localhost:8080/wmts/kvp?Service=WMTS&Request=GetTile&Version=1.0.0&Layer=remote.conc_chl&TileMatrix=0&TileRow=0&TileCol=0&Format=image/png>`_
    * `Get WMTS REST Capabilities (XML) <http://localhost:8080/wmts/1.0.0/WMTSCapabilities.xml>`_
    * `Get WMTS REST local tile (PNG) <http://localhost:8080/wmts/1.0.0/tile/local/conc_chl/0/0/1.png>`_
    * `Get WMTS REST remote tile (PNG) <http://localhost:8080/wmts/1.0.0/tile/remote/conc_chl/0/0/1.png>`_
* Tiles
    * `Get tile (PNG) <http://localhost:8080/datasets/local/vars/conc_chl/tiles/0/1/0.png>`_
    * `Get tile grid for OpenLayers 4.x <http://localhost:8080/datasets/local/vars/conc_chl/tilegrid?tiles=ol4>`_
    * `Get tile grid for Cesium 1.x <http://localhost:8080/datasets/local/vars/conc_chl/tilegrid?tiles=cesium>`_
    * `Get legend for layer (PNG) <http://localhost:8080/datasets/local/vars/conc_chl/legend.png>`_
* Time series service (preliminary & unstable, will likely change soon)
    * `Get time stamps per dataset <http://localhost:8080/ts>`_
    * `Get time series for single point <http://localhost:8080/ts/local/conc_chl/point?lat=51.4&lon=2.1&startDate=2017-01-15&endDate=2017-01-29>`_
* Places service (preliminary & unstable, will likely change soon>`_
    * `Get all features <http://localhost:8080/places/all>`_
    * `Get all features of collection "inside-cube" <http://localhost:8080/features/inside-cube>`_
    * `Get all features for dataset "local" <http://localhost:8080/places/all/local>`_
    * `Get all features of collection "inside-cube" for dataset "local" <http://localhost:8080/places/inside-cube/local>`_


xcube Viewer
============

xcube datasets published through ``xcube serve`` can be visualised using the `xcube-viewer <https://github.com/dcs4cop/xcube-viewer/>`_
web application.
To do so, run ``xcube serve`` with the ``--show`` flag.

In order make this option usable, xcube-viewer must be installed and build:

1. Download and install `yarn <https://yarnpkg.com/lang/en/>`_.

2. Download and build xcube-viewer:

::

    $ git clone https://github.com/dcs4cop/xcube-viewer.git
    $ cd xcube-viewer
    $ yarn build

3. Configure ``xcube serve`` so it finds the xcube-viewer
   On Linux (please adjust path):

::

    $ export XCUBE_VIEWER_PATH=/abs/path/to/xcube-viewer/build

   On Windows (please adjust path):

::

    > SET XCUBE_VIEWER_PATH=/abs/path/to/xcube-viewer/build

4. Then run ``xcube serve --show``:

::

    $ xcube serve --show --styles conc_chl=(0,20,"viridis") examples/serve/demo/cube-1-250-250.zarr

Viewing the generated xcube dataset described in the example :doc:``xcube_gen``:

::

    $ xcube serve --show --styles "analysed_sst=(200,375,'plasma')" demo_SST_xcube-optimized.zarr


In case you get an error message "cannot reach server" on the very bottom of the web app's main window,
refresh the page.

You can play around with the value range displayed in the viewer, here it is set to min=200K and max=375K.
The colormap used for mapping can be modified as well and the
`colormaps provided by matplotlib <https://matplotlib.org/examples/color/colormaps_reference.html>`_ can be used.


Other clients
=============

There are example HTML pages for some tile server clients. They need to be run in
a web server. If you don't have one, you can use Node's ``httpserver``:

::

    $ npm install -g httpserver

After starting both the xcube server and web server, e.g. on port 9090:

::

    $ httpserver -d -p 9090

you can run the client demos by following their links given below.


OpenLayers
----------

* `OpenLayers 4 Demo <http://localhost:9090/examples/serve/demo/index-ol4.html>`_
* `OpenLayers 4 Demo with WMTS <http://localhost:9090/examples/serve/demo/index-ol4-wmts.html>`_

Cesium
------

To run the `Cesium Demo <http://localhost:9090/examples/serve/demo/index-cesium.html>`_ first
`download Cesium <https://cesiumjs.org/downloads/>`_ and unpack the zip
into the ``xcube serve`` source directory so that there exists an
``./Cesium-x.y.z`` sub-directory. You may have to adapt the Cesium version number
in the `demo's HTML file <https://github.com/dcs4cop/xcube/blob/master/examples/serve/demo/index-cesium.html>`_.

