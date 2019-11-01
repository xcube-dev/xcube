.. _xcube.core.gen.iproc.DefaultInputProcessor: https://github.com/dcs4cop/xcube/blob/master/xcube/core/gen/iproc.py

=======
Plugins
=======

xcube's functionality can be extended by plugins. A plugin is a Python module that is detected
by xcube at runtime and dynamically loaded on demand.

Available Plugins
=================

*Coming soon...*


Plugin Development
==================

When a plugin is loaded, it adds its extensions to predefined *extension points* defined by xcube.
xcube defines the following extension points:

* ``xcube.core.gen.iproc``: input processor extensions
* ``xcube.core.dsio``: dataset I/O extensions
* ``xcube.cli``: Command-line interface (CLI) extensions


Input Processor Extensions
--------------------------

Input processors are used the ``xcube gen`` CLI command and ``gen_cube`` API function.
An input processor is responsible for processing individual time slices after they have been
opened from their sources and before they are appended to or inserted into the data cube
to be generated.

By default, xcube uses a standard input processor named ``default`` that expects inputs
to be individual NetCDF files that conform to the CF-convention. Every file is expected
to contain a single spatial image with dimensions ``lat`` and ``lon`` and the time
is expected to be given as global attributes.

If your input files do not conform with the ``default`` expectations, you can extend xcube
and write your own input processor. An input processor is an implementation of the
:class:`xcube.core.gen.iproc.InputProcessor` or :class:`xcube.core.gen.iproc.XYInputProcessor`
class.

As an example take a look at the implementation of the ``default`` input processor
`xcube.core.gen.iproc.DefaultInputProcessor`_.

*More coming soon...*

Dataset I/O Extensions
----------------------

*More coming soon...*

CLI Extensions
--------------

*More coming soon...*
