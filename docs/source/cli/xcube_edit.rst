.. _`demo cube-1-250-250.zarr`:  https://github.com/dcs4cop/xcube/tree/master/examples/serve/demo/cube-1-250-250.zarr
.. _`examples folder`: https://github.com/dcs4cop/xcube/tree/master/examples/edit/edit_metadata_cube-1-250-250.yml

==================
``xcube edit``
==================

Synopsis
========

Optimize xcube dataset for faster access.

::

    $ xcube edit --help

::

    Usage: xcube edit [OPTIONS] CUBE

      Edit the metadata of an xcube dataset. Edits the metadata of a given CUBE.
      The command currently works only for data cubes using ZARR format.

    Options:
      -o, --output OUTPUT      Output path. The placeholder "<built-in function
                               input>" will be replaced by the input's filename
                               without extension (such as ".zarr"). Defaults to
                               "{input}-edited.zarr".
      -M, --metadata METADATA  The metadata of the cube is edited. The metadata to
                               be changed should be passed over in a single yml
                               file.
      -I, --in-place           Edit the cube in place. Ignores output path.
      --help                   Show this message and exit.


Examples
========

The global attributes of the demo xcube dataset  `demo cube-1-250-250.zarr`_ do not contain the creators name
not an url. Furthermore the long name of the variable 'conc_chl' is 'Chlorophylll concentration', with too many l's.
This can be fixed by using xcube edit. A yml-file defining the key words to be changed with the new content has to
be created. The demo yml is saved in the `examples folder`_.

Edit the metadata of the existing xcube dataset  ``cube-1-250-250-edited.zarr``:


::

    $ xcube edit /examples/serve/demo/cube-1-250-250.zarr -M examples/edit/edit_metadata_cube-1-250-250.yml -o cube-1-250-250-edited.zarr
    


Python API
==========

The related Python API function is :py:func:`xcube.api.edit_metadata`.
