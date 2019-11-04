.. _xcube.core.gen.iproc.DefaultInputProcessor: https://github.com/dcs4cop/xcube/blob/master/xcube/core/gen/iproc.py

=======
Plugins
=======

xcube's functionality can be extended by plugins. A plugin is a Python module that is detected
by xcube at runtime and dynamically loaded by xcube on demand, that is, once the extensions
added by plugins need to be made available.

Installing Plugins
==================

Plugins are installed by simply installing the plugin's package into xcube's Python environment.

In order to be detected by xcube, an plugin package's name must either start with ``xcube_``
or the plugin package's ``setup.py`` file must specify an entry point in the group
``xcube_plugins``. Details are provided below in section `plugin_development`_.


Available Plugins
=================

SENTINEL Hub
------------

The [xcube-sh](https://github.com/dcs4cop/xcube-sh) plugin adds support for the
[SENTINEL Hub Cloud API](https://www.sentinel-hub.com/). It extends xcube by
* a new Python API function ``xcube_sh.open.open_cube`` to create data cubes from SENTINEL Hub on-the-fly;
* adding a new CLI command ``xcube sh`` to write data cubes created from SENTINEL Hub into the file system.

Cube Generation
---------------

xcube's GitHub organisation currently hosts a few plugins that add new *input processor* extensions
(see below) to xcube's data cube generation tool :doc:`cli/xcube_gen`. They are very specific
but are a good starting point for developing your own input processors:

* [xcube-gen-bc](https://github.com/dcs4cop/xcube-gen-bc) - adds new input processors for specific
  Ocean Colour Earth Observation products derived from the Sentinel-3 OLCI measurements.
* [xcube-gen-rbins](https://github.com/dcs4cop/xcube-gen-rbins) - adds new input processors for specific
  Ocean Colour Earth Observation products derived from the SEVIRI measurements.
* [xcube-gen-vito](https://github.com/dcs4cop/xcube-gen-vito) - adds new input processors for specific
  Ocean Colour Earth Observation products derived from the Sentinel-2 MSI measurements.


.. _plugin_development:

Plugin Development
==================

Plugin Definition
-----------------

An xcube plugin is a Python package that is installed in xcube's Python environment.
xcube can detect plugins either

1. by naming convention (more simple);
2. by entry point (more flexible).

By naming convention: Any Python package named ``xcube_<name>`` that defines a plugin *initializer function*
named ``init_plugin`` either defined in ``xcube_<name>/plugin.py`` (preferred) or ``xcube_<name>/__init__.py``
is an xcube plugin.

By entry point: Any Python package installed using [Setuptools](https://setuptools.readthedocs.io/) that
defines a non-empty entry point group ``xcube_plugins`` is an xcube plugin. An entry point in the
``xcube_plugins`` group has the format ``<name> = <fully-qualified-module-path>:<init-func-name>``,
and therefore specifies where plugin *initializer function* named ``<init-func-name>`` is found.
As an example, refer to the xcube standard plugin definitions in xcube's
[``setup.py``](https://github.com/dcs4cop/xcube/blob/master/setup.py) file.

For more information on Setuptools entry points refer to section
[Dynamic Discovery of Services and Plugins](https://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins)
in the [Setuptools documentation](https://setuptools.readthedocs.io/).


Initializer Function
--------------------

xcube plugins are initialized using a dedicated function that has a single *extension registry* argument,
that is used by plugins's to register their extensions to xcube. By convention, this function is called
``init_plugin``, however, when using entry points, it can have any name. As an example, here is the initializer
function of the SENTINEL Hub plugin ``xcube_sh/plugin.py``:::


    from xcube.constants import EXTENSION_POINT_CLI_COMMANDS
    from xcube.util import extension


    def init_plugin(ext_registry: extension.ExtensionRegistry):
        """xcube SentinelHub extensions"""
        ext_registry.add_extension(loader=extension.import_component('xcube_sh.cli:cli'),
                                   point=EXTENSION_POINT_CLI_COMMANDS,
                                   name='sh_cli')


Extension Points and Extensions
-------------------------------

When a plugin is loaded, it adds its extensions to predefined *extension points* defined by xcube.
xcube defines the following extension points:

* ``xcube.core.gen.iproc``: input processor extensions
* ``xcube.core.dsio``: dataset I/O extensions
* ``xcube.cli``: Command-line interface (CLI) extensions

An extension is added to the extension registry's ``add_extension`` method. The extension registry is
passed to the plugin initializer function as its only argument.


Input Processor Extensions
--------------------------

Input processors are used the ``xcube gen`` CLI command and ``gen_cube`` API function.
An input processor is responsible for processing individual time slices after they have been
opened from their sources and before they are appended to or inserted into the data cube
to be generated. New input processors are usually programmed to support the characteristics
of specific ``xcube gen`` inputs, mostly specific Earth Observation data products.

By default, xcube uses a standard input processor named ``default`` that expects inputs
to be individual NetCDF files that conform to the CF-convention. Every file is expected
to contain a single spatial image with dimensions ``lat`` and ``lon`` and the time
is expected to be given as global attributes.

If your input files do not conform with the ``default`` expectations, you can extend xcube
and write your own input processor. An input processor is an implementation of the
:class:`xcube.core.gen.iproc.InputProcessor` or :class:`xcube.core.gen.iproc.XYInputProcessor`
class.

As an example take a look at the implementation of the ``default`` input processor
`xcube.core.gen.iproc.DefaultInputProcessor`_ or the various input processor plugins mentioned above.

The extension point identifier is defined by the constant ``xcube.constants.EXTENSION_POINT_INPUT_PROCESSORS``.

Dataset I/O Extensions
----------------------

*More coming soon...*

The extension point identifier is defined by the constant ``xcube.constants.EXTENSION_POINT_DATASET_IOS``.

CLI Extensions
--------------

*More coming soon...*

The extension point identifier is defined by the constant ``xcube.constants.EXTENSION_POINT_CLI_COMMANDS``.
