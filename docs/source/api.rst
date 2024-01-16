==========
Python API
==========

Data Store Framework
====================

Functions
---------

.. autofunction:: xcube.core.store.new_data_store

.. autofunction:: xcube.core.store.new_fs_data_store

.. autofunction:: xcube.core.store.find_data_store_extensions

.. autofunction:: xcube.core.store.get_data_store_class

.. autofunction:: xcube.core.store.get_data_store_params_schema

Classes
-------

.. autoclass:: xcube.core.store.DataStore
    :members:

.. autoclass:: xcube.core.store.MutableDataStore
    :members:

.. autoclass:: xcube.core.store.DataOpener
    :members:

.. autoclass:: xcube.core.store.DataSearcher
    :members:

.. autoclass:: xcube.core.store.DataWriter
    :members:

.. autoclass:: xcube.core.store.DataStoreError
    :members:

.. autoclass:: xcube.core.store.DataDescriptor
    :members:

.. autoclass:: xcube.core.store.DatasetDescriptor
    :members:

.. autoclass:: xcube.core.store.MultiLevelDatasetDescriptor
    :members:

.. autoclass:: xcube.core.store.DatasetDescriptor
    :members:

.. autoclass:: xcube.core.store.VariableDescriptor
    :members:

.. autoclass:: xcube.core.store.GeoDataFrameDescriptor
    :members:

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

Cube Resampling
===============

.. autofunction:: xcube.core.resampling.affine_transform_dataset

.. autofunction:: xcube.core.resampling.resample_ndimage

.. autofunction:: xcube.core.resampling.encode_grid_mapping

.. autofunction:: xcube.core.resampling.rectify_dataset

.. autofunction:: xcube.core.resampling.resample_in_space

.. autofunction:: xcube.core.resampling.resample_in_time

Cube Manipulation
=================

.. autofunction:: xcube.core.vars2dim.vars_to_dim

.. autofunction:: xcube.core.chunk.chunk_dataset

.. autofunction:: xcube.core.unchunk.unchunk_dataset

.. autofunction:: xcube.core.optimize.optimize_dataset

Cube Subsetting
===============

.. autofunction:: xcube.core.select.select_variables_subset

.. autofunction:: xcube.core.geom.clip_dataset_by_geometry


Cube Masking
============

.. autofunction:: xcube.core.geom.mask_dataset_by_geometry

.. autoclass:: xcube.core.maskset.MaskSet
    :members:


Rasterisation of Features
=========================

.. autofunction:: xcube.core.geom.rasterize_features


Cube Metadata
=============

.. autofunction:: xcube.core.update.update_dataset_attrs

.. autofunction:: xcube.core.update.update_dataset_spatial_attrs

.. autofunction:: xcube.core.update.update_dataset_temporal_attrs


Cube verification
=================

.. autofunction:: xcube.core.verify.assert_cube

.. autofunction:: xcube.core.verify.verify_cube

Multi-Resolution Datasets
=========================

.. autoclass:: xcube.core.mldataset.MultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.BaseMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.CombinedMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.ComputedMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.FsMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.IdentityMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.LazyMultiLevelDataset
    :members:

.. autoclass:: xcube.core.mldataset.MappedMultiLevelDataset
    :members:

Zarr Store
==========

.. autoclass:: xcube.core.zarrstore.ZarrStoreHolder
    :members:

.. autoclass:: xcube.core.zarrstore.GenericZarrStore
    :members:

.. autoclass:: xcube.core.zarrstore.GenericArray
    :members:

.. autoclass:: xcube.core.zarrstore.CachedZarrStore
    :members:

.. autoclass:: xcube.core.zarrstore.DiagnosticZarrStore
    :members:

Utilities
=========

.. autoclass:: xcube.core.gridmapping.GridMapping
    :members:

.. autofunction:: xcube.core.geom.convert_geometry

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
