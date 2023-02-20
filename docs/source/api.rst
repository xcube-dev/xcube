==========
Python API
==========

Cube I/O
========

.. autofunction:: xcube.core.dsio.open_cube

.. autofunction:: xcube.core.dsio.write_cube

Cube generation
===============

.. autofunction:: xcube.core.gen.gen.gen_cube

.. autofunction:: xcube.core.new.new_cube

Cube computation
================

.. autofunction:: xcube.core.compute.compute_cube

.. autofunction:: xcube.core.evaluate.evaluate_dataset

Cube data extraction
====================

.. autofunction:: xcube.core.extract.get_cube_values_for_points

.. autofunction:: xcube.core.extract.get_cube_point_indexes

.. autofunction:: xcube.core.extract.get_cube_values_for_indexes

.. autofunction:: xcube.core.extract.get_dataset_indexes

.. autofunction:: xcube.core.timeseries.get_time_series

Cube manipulation
=================

.. autofunction:: xcube.core.resample.resample_in_time

.. autofunction:: xcube.core.vars2dim.vars_to_dim

.. autofunction:: xcube.core.chunk.chunk_dataset

.. autofunction:: xcube.core.unchunk.unchunk_dataset

.. autofunction:: xcube.core.optimize.optimize_dataset

Cube subsetting
===============

.. autofunction:: xcube.core.select.select_variables_subset

.. autofunction:: xcube.core.geom.clip_dataset_by_geometry


Cube masking
============

.. autofunction:: xcube.core.geom.mask_dataset_by_geometry

.. autoclass:: xcube.core.maskset.MaskSet
    :members:


Rasterisation of Features
=========================

.. autofunction:: xcube.core.geom.rasterize_features


Cube metadata
=============

.. autofunction:: xcube.core.edit.edit_metadata

.. autofunction:: xcube.core.update.update_dataset_attrs

.. autofunction:: xcube.core.update.update_dataset_spatial_attrs

.. autofunction:: xcube.core.update.update_dataset_temporal_attrs


Cube verification
=================

.. autofunction:: xcube.core.verify.assert_cube

.. autofunction:: xcube.core.verify.verify_cube

Multi-resolution pyramids
=========================

.. autofunction:: xcube.core.level.compute_levels

.. autofunction:: xcube.core.level.read_levels

.. autofunction:: xcube.core.level.write_levels

Utilities
=========

.. autofunction:: xcube.core.geom.convert_geometry

.. autoclass:: xcube.core.store.CubeStore
    :members:

.. autoclass:: xcube.core.schema.CubeSchema
    :members:

.. autofunction:: xcube.util.dask.new_cluster

Plugin Development
==================

.. autoclass:: xcube.util.extension.ExtensionRegistry
    :members:

.. autoclass:: xcube.util.extension.Extension
    :members:

.. autofunction:: xcube.util.extension.import_component

.. autodata:: xcube.constants.EXTENSION_POINT_INPUT_PROCESSORS

.. autodata:: xcube.constants.EXTENSION_POINT_DATASET_IOS

.. autodata:: xcube.constants.EXTENSION_POINT_CLI_COMMANDS

.. autofunction:: xcube.util.plugin.get_extension_registry

.. autofunction:: xcube.util.plugin.get_plugins
