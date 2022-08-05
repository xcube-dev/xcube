
===============
``xcube patch``
===============

Synopsis
========

Patch and consolidate the metadata of a xcube dataset.

::

    $ xcube patch --help

::

    Usage: xcube patch [OPTIONS] DATASET

      Patch and consolidate the metadata of a dataset.

      DATASET can be either a local filesystem path or a URL. It must point to
      either a Zarr dataset (*.zarr) or a xcube multi-level dataset (*.levels).
      Additional storage options for a given protocol may be passed by the OPTIONS
      option.

      In METADATA, the special attribute value "__delete__" can be used to remove
      that attribute from dataset or array metadata.

    Options:
      --metadata METADATA  The metadata to be patched. Must be a JSON or YAML file
                           using Zarr consolidated metadata format.
      --options OPTIONS    Protocol-specific storage options (see fsspec). Must be
                           a JSON or YAML file.
      -q, --quiet          Disable output of log messages to the console entirely.
                           Note, this will also suppress error and warning
                           messages.
      -v, --verbose        Enable output of log messages to the console. Has no
                           effect if --quiet/-q is used. May be given multiple
                           times to control the level of log messages, i.e., -v
                           refers to level INFO, -vv to DETAIL, -vvv to DEBUG,
                           -vvvv to TRACE. If omitted, the log level of the
                           console is WARNING.
      -d, --dry-run        Do not change any data, just report what would have
                           been changed.
      --help               Show this message and exit.



Patch file example
==================

Patch files use the Zarr Consolidated Metadata Format, v1.
For example, the following patch file (YAML) will delete the 
global attribute ``TileSize`` and change the value of the 
attribute ``long_name`` of variable ``conc_chl``:

.. code-block:: yaml

    zarr_consolidated_format: 1
    metadata:

      .zattrs:
        TileSize: __delete__

      conc_chl/.zattrs:
        long_name: Chlorophyll concentration


Storage options file example
============================

Here is a storage options file for the "s3" protocol that
provides credentials for AWS S3 access:

.. code-block:: yaml

    key: AJDKJCLSKKA
    secret: kjkl456lkj45632k45j63l


Usage example
=============

.. code-block:: bash

    $ xcube patch s3://my-cubes-bucket/test.zarr --metadata patch.yml -v
