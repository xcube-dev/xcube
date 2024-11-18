===
CLI
===

The xcube command-line interface (CLI) is a single executable :doc:`cli/xcube` with several
sub-commands comprising functions ranging from xcube dataset generation, over analysis and
manipulation, to dataset publication.

Common Arguments and Options
============================

Most of the commands operate on inputs that are xcube datasets. Such inputs are consistently named
``CUBE`` and provided as one or more command arguments. CUBE inputs may be a path into the
local file system or a path into some object storage bucket, e.g. in AWS S3.
Command inputs of other types are consistently called ``INPUT``.

Many commands also output something, i.e. are writing files. The paths or names of such outputs are
consistently provided by the ``-o OUTPUT`` or ``--output OUTPUT`` option. As the output is an option,
there is usually a default value for it. If multiply file formats are supported, commands usually
provide a ``-f FORMAT`` or ``--format FORMAT`` option. If omitted, the format may be guessed from the
output's name.

Cube generation
===============

.. toctree::
   :maxdepth: 1

   cli/xcube_gen
   cli/xcube_grid

Cube computation
================

.. toctree::
   :maxdepth: 1

   cli/xcube_compute

Cube inspection
===============

.. toctree::
   :maxdepth: 1

   cli/xcube_dump
   cli/xcube_verify

Cube data extraction
====================

.. toctree::
   :maxdepth: 1

   cli/xcube_extract

Cube manipulation
=================

.. toctree::
   :maxdepth: 1

   cli/xcube_chunk
   cli/xcube_level
   cli/xcube_optimize
   cli/xcube_patch
   cli/xcube_prune
   cli/xcube_resample
   cli/xcube_vars2dim

Cube conversion
===============

.. toctree::
   :maxdepth: 1

   cli/xcube_level

Cube publication
================

.. toctree::
   :maxdepth: 1

   cli/xcube_serve
