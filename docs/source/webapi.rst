.. _`WMTS`: https://en.wikipedia.org/wiki/Web_Map_Tile_Service


==================
Web API and Server
==================

xcube's RESTful web API is used to publish data cubes to clients. Using the
API, clients can

* List configured xcube datasets;
* Get xcube dataset details including metadata, coordinate data, and metadata
  about all included variables;
* Get cube data;
* Extract time-series statistics from any variable given any geometry;
* Get spatial image tiles from any variable;
* Browse datasets and retrieve dataset data and metadata using the STAC API;
* Get places (GeoJSON features including vector data) that can be associated
  with xcube datasets;
* Perform compute operations on datasets, with the results calculated on
  demand and presented as new, dynamically generated datasets;
* Retrieve coverages backed by datasets using the OGC API - Coverages API.

Later versions of API will also allow for xcube dataset management including
generation, modification, and deletion of xcube datasets.

The complete description of all available functions is provided via
openapi.html after starting the server locally. Please check out
:doc:`examples/xcube_serve` to learn how to do access it.

The web API is provided through the *xcube server* which is started using the
:doc:`cli/xcube_serve` CLI command.
