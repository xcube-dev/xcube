.. _CF conventions: http://cfconventions.org/cf-conventions/cf-conventions.html
.. _`dask`: https://dask.readthedocs.io/
.. _xarray: http://xarray.pydata.org/
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/data-structures.html#dataset
.. _xarray.DataArray: http://xarray.pydata.org/en/stable/data-structures.html#dataarray
.. _`zarr`: https://zarr.readthedocs.io/
.. _`Zarr format`: https://zarr.readthedocs.io/en/stable/spec/v2.html
.. _`Sentinel Hub`: https://www.sentinel-hub.com/

==========
Data cube?
==========

The interpretation of the term *data cube* in the Earth observation (EO) domain usually depends
on the current context. It may refer to a data service such as `Sentinel Hub`_, to some abstract
API, or to a concrete set of spatial images that form a time-series.

This chapter briefly explains the very specific data cube concept used in the xcube project.
For details, please refer to the :doc:`cubespec`.

Data Model
==========

An xcube data cube comprises one or more (geo-physical) data variables
whose values are stored in cells of a common multi-dimensional, spatio-temporal grid.
The dimensions are usually time, latitude, and longitude, however other dimensions may be present.

All xcube data cubes are structured in the same way following a common data model.
They are also self-describing by providing metadata for the cube and
all cube's variables following the `CF conventions`_.

A cube's in-memory representation in Python programs is an `xarray.Dataset`_ instance. Each data variable is
represented by multi-dimensional `xarray.DataArray`_ that is arranged in non-overlapping, contiguous
sub-regions (data chunks). The data chunks allow for out-of-core computation of cube dataset's that don't fit
in a single computer's RAM.

Processing Model
================

When data cubes are opened, only the cube's structure and its metadata are loaded into memory. The actual
data arrays of variables are loaded on-demand only, and only for chunks intersecting the desired sub-region.

Operations that generate new data variables from existing ones will be chunked
in the same way. Therefore such operation chains generate a processing graph providing a deferred, concurrent
execution model.

Data Format
===========

For the external, physical representation of cube we usually use the `Zarr format`_ that supports parallel
processing data chunks that may be fetched from remote cloud storage such as S3 and GCS.


Python Packages
===============

The in-memory data model, processing model, and the external format used in xcube
are provided through the excellent Python `xarray`_, `dask`_, and `zarr`_ packages.
