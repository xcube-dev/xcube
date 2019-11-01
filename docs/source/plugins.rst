.. _xarray: http://xarray.pydata.org/

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

An input processor is an implementation of the ``xcube.api.gen.iproc.InputProcessor``
class.

As an example look into the standards have a look

*Coming soon...*

Dataset I/O Extensions
----------------------

*Coming soon...*

CLI Extensions
--------------

*Coming soon...*