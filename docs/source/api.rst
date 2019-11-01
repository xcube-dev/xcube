==========
Python API
==========

Cube I/O
========

.. autofunction:: xcube.api.read_cube

.. autofunction:: xcube.api.open_cube

.. autofunction:: xcube.api.write_cube

Cube generation
===============

.. autofunction:: xcube.api.gen_cube

.. autofunction:: xcube.api.new_cube


Cube data extraction
====================

.. autofunction:: xcube.api.get_cube_values_for_points

.. autofunction:: xcube.api.get_cube_point_indexes

.. autofunction:: xcube.api.get_cube_values_for_indexes

.. autofunction:: xcube.api.get_dataset_indexes

.. autofunction:: xcube.api.get_time_series

Cube manipulation
=================

.. autofunction:: xcube.api.resample_in_time

.. autofunction:: xcube.api.vars_to_dim

.. autofunction:: xcube.api.chunk_dataset

.. autofunction:: xcube.api.unchunk_dataset

.. autofunction:: xcube.api.vars_to_dim

Cube subsetting
===============

.. autofunction:: xcube.api.select_vars

.. autofunction:: xcube.api.clip_dataset_by_geometry


Cube masking
============

.. autofunction:: xcube.api.mask_dataset_by_geometry

.. autoclass:: xcube.api.MaskSet


Cube optimization
=================

.. autofunction:: xcube.api.optimize_dataset


Cube metadata
=============

.. autofunction:: xcube.api.edit_metadata

.. autofunction:: xcube.api.update_dataset_attrs

.. autofunction:: xcube.api.update_dataset_spatial_attrs

.. autofunction:: xcube.api.update_dataset_temporal_attrs


Cube verification
=================

.. autofunction:: xcube.api.assert_cube

.. autofunction:: xcube.api.verify_cube

Multi-resolution pyramids
=========================

.. autofunction:: xcube.api.compute_levels

.. autofunction:: xcube.api.read_levels

.. autofunction:: xcube.api.write_levels

Utilities
=========

.. autofunction:: xcube.api.convert_geometry
