.. _CF conventions: http://cfconventions.org/cf-conventions/cf-conventions.html
.. _`dask`: https://dask.readthedocs.io/
.. _`JupyterLab`: https://jupyterlab.readthedocs.io/
.. _xarray: http://xarray.pydata.org/
.. _xarray API: http://xarray.pydata.org/en/stable/api.html
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/data-structures.html#dataset
.. _xarray.DataArray: http://xarray.pydata.org/en/stable/data-structures.html#dataarray
.. _`zarr`: https://zarr.readthedocs.io/
.. _`Zarr format`: https://zarr.readthedocs.io/en/stable/spec/v2.html
.. _`Sentinel Hub`: https://www.sentinel-hub.com/

========
Overview
========

xcube is an open-source Python package and toolkit that has been developed to provide Earth observation (EO) data in an
analysis-ready form to users. We do this by carefully converting EO data sources into self-contained *data cubes*
that can be published in the cloud.

What is a Data Cube?
====================

The interpretation of the term *data cube* in the EO domain usually depends
on the current context. It may refer to a data service such as `Sentinel Hub`_, to some abstract
API, or to a concrete set of spatial images that form a time-series.

This section briefly explains the specific concept of a data cube used in the xcube project - the xcube dataset.

Data Model
----------

An xcube dataset comprises one or more (geo-physical) data variables
whose values are stored in cells of a common multi-dimensional, spatio-temporal grid.
The dimensions are usually time, latitude, and longitude, however other dimensions may be present.

All xcube datasets are structured in the same way following a common data model.
They are also self-describing by providing metadata for the cube and
all cube's variables following the `CF conventions`_.
For details regarding the common data model, please refer to the :doc:`cubespec`.

A xcube dataset's in-memory representation in Python programs is an `xarray.Dataset`_ instance. Each
dataset variable is represented by multi-dimensional `xarray.DataArray`_ that is arranged in non-overlapping,
contiguous sub-regions (data chunks). The data chunks allow for out-of-core computation of cube dataset's that don't
fit in a single computer's RAM.

Processing Model
----------------

When xcube datasets are opened, only the cube's structure and its metadata are loaded into memory. The actual
data arrays of variables are loaded on-demand only, and only for chunks intersecting the desired sub-region.

Operations that generate new data variables from existing ones will be chunked
in the same way. Therefore such operation chains generate a processing graph providing a deferred, concurrent
execution model.

Data Format
-----------

For the external, physical representation of cube we usually use the `Zarr format`_ that supports parallel
processing data chunks that may be fetched from remote cloud storage such as S3 and GCS.

Python Packages
---------------

The xcube package builds heavily on Python’s big data ecosystem for handling huge N-dimensional data arrays
and exploiting cloud-based storage and processing resources. In particular, xcube's in-memory data model is
provided by `xarray`_, the memory management and processing model is provided through `dask`_,
and the external format is provided by `zarr`_. xarray, dask, and zarr have increased their popularity for
big data solutions over the last couple of years, for creating a scalable and efficient EO data solutions.

The Tools
=========

On top of `xarray`_, `dask`_, `zarr`_, and other popular Python data science packages,
xcube provides various higher-level tools to generate, manipulate, and publish xcube datasets:

* :doc:`cli` -access, generate, modify, and analyse xcube datasets;
* :doc:`api` - access, generate, modify, and analyse xcube datasets via Python programs and notebooks;
* :doc:`webapi` - access, generate, modify, and analyse xcube datasets via a RESTful API;
* :doc:`viewer` – publish and visualise xcube datasets using maps and time-series charts.

A typical workflow:

1. generate an xcube dataset from some EO data sources
   using the :doc:`cli/xcube_gen` tool;
2. optimize the xcube dataset with respect to specific use cases
   using the :doc:`cli/xcube_optimize` and :doc:`cli/xcube_prune` tools.
3. deploy xcube datasets to some accessible location (e.g. on AWS S3).

Then users can

4. access, analyse, modify, transform, visualise the data using the :doc:`api` and `xarray API`_ through
   Python programs or `JupyterLab`_.

Another way to let users interact with the data is to

5. publish the xcube datasets through a web API
   using the :doc:`cli/xcube_serve` tool;
6. visualise the xcube datasets by
   using :doc:`viewer`.

