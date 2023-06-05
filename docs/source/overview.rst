.. _CF conventions: http://cfconventions.org/cf-conventions/cf-conventions.html
.. _`dask`: https://dask.readthedocs.io/
.. _`JupyterLab`: https://jupyterlab.readthedocs.io/
.. _`WMTS`: https://en.wikipedia.org/wiki/Web_Map_Tile_Service
.. _xarray: http://xarray.pydata.org/
.. _xarray API: http://xarray.pydata.org/en/stable/api.html
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/data-structures.html#dataset
.. _xarray.DataArray: http://xarray.pydata.org/en/stable/data-structures.html#dataarray
.. _`zarr`: https://zarr.readthedocs.io/
.. _`Zarr format`: https://zarr.readthedocs.io/en/stable/spec/v2.html
.. _`Sentinel Hub`: https://www.sentinel-hub.com/
.. _`Chunking and Performance`: http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance

========
Overview
========

*xcube* is an open-source Python package and toolkit that has been developed to provide Earth observation (EO) data in an
analysis-ready form to users. xcube achieves this by carefully converting EO data sources into self-contained *data cubes*
that can be published in the cloud. The Python package is expanded and maintained by
[Brockmann Consult GmbH](https://www.brockmann-consult.de)

Data Cube
=========

The interpretation of the term *data cube* in the EO domain usually depends
on the current context. It may refer to a data service such as `Sentinel Hub`_, to some abstract
API, or to a concrete set of spatial images that form a time-series.

This section briefly explains the specific concept of a data cube used in the xcube project - the *xcube dataset*.

xcube Dataset
=============

Data Model
----------

An xcube dataset contains one or more (geo-physical) data variables
whose values are stored in cells of a common multi-dimensional, spatio-temporal grid.
The dimensions are usually time, latitude, and longitude, however other dimensions may be present.

All xcube datasets are structured in the same way following a common data model.
They are also self-describing by providing metadata for the cube and
all cube's variables following the `CF conventions`_.
For details regarding the common data model, please refer to the :doc:`cubespec`.

A xcube dataset's in-memory representation in Python programs is an `xarray.Dataset`_ instance. Each
dataset variable is represented by multi-dimensional `xarray.DataArray`_ that is arranged in non-overlapping,
contiguous sub-regions called *data chunks*.

Data Chunks
-----------

Chunked variables allow for out-of-core computations of xcube dataset that don't fit in a single computer's RAM as
data chunks can be processed independently from each other.

The way how dataset variables are sub-divided into smaller chunks - their *chunking* -
has a substantial impact on processing performance and there is no single ideal
chunking for all use cases. For time series analyses it is preferable to have chunks with a
smaller spatial dimensions and larger time dimension, for spatial analyses and visualisation
on using a map, the opposite is the case.

xcube provide tools for re-chunking of xcube datasets (:doc:`cli/xcube_chunk`, :doc:`cli/xcube_level`)
and the xcube server (:doc:`cli/xcube_serve`) allows
serving the same data cubes using different chunkings. For further reading have a look into the
`Chunking and Performance`_ section of the xarray documentation.

Processing Model
----------------

When xcube datasets are opened, only the cube's structure and its metadata are loaded into memory. The actual
data arrays of variables are loaded on-demand only, and only for chunks intersecting the desired sub-region.

Operations that generate new data variables from existing ones will be chunked
in the same way. Therefore, such operation chains generate a processing graph providing a deferred, concurrent
execution model.

Data Format
-----------

For the external, physical representation of xcube datasets we usually use the `Zarr format`_. Zarr takes full
advantage of data chunks and supports parallel processing of chunks that may originate from the local file system
or from remote cloud storage such as S3 and GCS.

Python Packages
---------------

The xcube package builds heavily on Python’s big data ecosystem for handling huge N-dimensional data arrays
and exploiting cloud-based storage and processing resources. In particular, xcube's in-memory data model is
provided by `xarray`_, the memory management and processing model is provided through `dask`_,
and the external format is provided by `zarr`_. xarray, dask, and zarr have increased their popularity for
big data solutions over the last couple of years, for creating scalable and efficient EO data solutions.

Toolkit
=======

On top of `xarray`_, `dask`_, `zarr`_, and other popular Python data science packages,
xcube provides various higher-level tools to generate, manipulate, and publish xcube datasets:

* :doc:`cli` - access, generate, modify, and analyse xcube datasets using the ``xcube`` tool;
* :doc:`api` - access, generate, modify, and analyse xcube datasets via Python programs and notebooks;
* :doc:`webapi` - access, analyse, visualize xcube datasets via an xcube server;
* :doc:`viewer` – publish and visualise xcube datasets using maps and time-series charts.


Workflows
=========

The basic use case is to generate an xcube dataset and deploy it so that your users can access it:

1. generate an xcube dataset from some EO data sources
   using the :doc:`cli/xcube_gen` tool with a specific *input processor*.
2. optimize the generated xcube dataset with respect to specific use cases
   using the :doc:`cli/xcube_chunk` tool.
3. optimize the generated xcube dataset by consolidating metadata and elimination of empty chunks
   using :doc:`cli/xcube_optimize` and :doc:`cli/xcube_prune` tools.
4. deploy the optimized xcube dataset(s) to some location (e.g. on AWS S3) where users can access them.

Then you can:

5. access, analyse, modify, transform, visualise the data using the :doc:`api` and `xarray API`_ through
   Python programs or `JupyterLab`_, or
6. extract data points by coordinates from a cube
   using the :doc:`cli/xcube_extract` tool, or
7. resample the cube in time to generate temporal aggregations
   using the :doc:`cli/xcube_resample` tool.

Another way to provide the data to users is via the *xcube server*, that provides a
RESTful API and a `WMTS`_. The latter is used
to visualise spatial subsets of xcube datasets efficiently at any zoom level.
To provide optimal visualisation and data extraction performance through the xcube server,
xcube datasets may be prepared beforehand. Steps 8 to 10 are optional.

8. verify a dataset to be published conforms with the :doc:`cubespec`
   using the :doc:`cli/xcube_verify` tool.
9. adjust your dataset chunking to be optimal for generating spatial image tiles and generate
   a multi-resolution image pyramid
   using the :doc:`cli/xcube_chunk` and :doc:`cli/xcube_level` tools.
10. create a dataset variant optimal for time series-extraction again
    using the :doc:`cli/xcube_chunk` tool.
11. configure xcube datasets and publish them through the xcube server
    using the :doc:`cli/xcube_serve` tool.

You may then use a WMTS-compatible client to visualise the datasets or develop your own xcube server client that
will make use of the xcube's REST API.

The easiest way to visualise your data is using the xcube :doc:`viewer`, a single-page web application that
can be configured to work with xcube server URLs.
