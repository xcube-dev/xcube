======================
Command-line Interface
======================

The xcube command-line interface is a single executable ``xcube`` with several sub-commands comprising
functions ranging from xcube dataset generation, over analysis and manipulation, to dataset publication.

Most of the commands operate on inputs that must be an (:doc:`cubespec`) - a data cube in the xcube sense.
Such inputs are consistently named ``CUBE`` and provided as command argument(s). You may pass a path into the
local file system or a path into some object storage bucket, e.g. in AWS S3.
Command inputs of other type are usually called ``INPUT``.

.. toctree::
   :caption: Main command
   :maxdepth: 1

   cli/xcube

.. toctree::
   :caption: Cube generation
   :maxdepth: 1

   cli/xcube_gen
   cli/xcube_grid

.. toctree::
   :caption: Cube inspection and data extraction
   :maxdepth: 1

   cli/xcube_dump
   cli/xcube_extract
   cli/xcube_verify

.. toctree::
   :caption: Cube manipulation
   :maxdepth: 1

   cli/xcube_chunk
   cli/xcube_level
   cli/xcube_optimize
   cli/xcube_prune
   cli/xcube_resample
   cli/xcube_vars2dim

.. toctree::
   :caption: Cube publication
   :maxdepth: 1

   cli/xcube_serve
