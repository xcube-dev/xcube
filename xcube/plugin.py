# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from xcube.constants import EXTENSION_POINT_CLI_COMMANDS
from xcube.constants import EXTENSION_POINT_DATASET_IOS
from xcube.constants import EXTENSION_POINT_INPUT_PROCESSORS
from xcube.constants import FORMAT_NAME_CSV
from xcube.constants import FORMAT_NAME_MEM
from xcube.constants import FORMAT_NAME_NETCDF4
from xcube.constants import FORMAT_NAME_ZARR
from xcube.util import extension


def init_plugin(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard extensions.
    """
    _register_input_processors(ext_registry)
    _register_dataset_ios(ext_registry)
    _register_cli_commands(ext_registry)


def _register_input_processors(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard input processors used by "xcube gen" and gen_cube().
    """
    ext_registry.add_extension(
        loader=extension.import_component('xcube.core.gen.iproc:DefaultInputProcessor'),
        point=EXTENSION_POINT_INPUT_PROCESSORS, name='default',
        description='Single-scene NetCDF/CF inputs in xcube standard format'
    )


def _register_dataset_ios(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard dataset I/O components used by various CLI and API functions.
    """
    ext_registry.add_extension(
        loader=extension.import_component('xcube.core.dsio:ZarrDatasetIO', call=True),
        point=EXTENSION_POINT_DATASET_IOS, name=FORMAT_NAME_ZARR,
        description='Zarr file format (http://zarr.readthedocs.io)',
        ext='zarr', modes={'r', 'w', 'a'}
    )
    ext_registry.add_extension(
        loader=extension.import_component('xcube.core.dsio:Netcdf4DatasetIO', call=True),
        point=EXTENSION_POINT_DATASET_IOS, name=FORMAT_NAME_NETCDF4,
        description='NetCDF-4 file format',
        ext='nc', modes={'r', 'w', 'a'}
    )
    ext_registry.add_extension(
        loader=extension.import_component('xcube.core.dsio:CsvDatasetIO', call=True),
        point=EXTENSION_POINT_DATASET_IOS, name=FORMAT_NAME_CSV,
        description='CSV file format',
        ext='csv', modes={'r', 'w'}
    )
    ext_registry.add_extension(
        loader=extension.import_component('xcube.core.dsio:MemDatasetIO', call=True),
        point=EXTENSION_POINT_DATASET_IOS, name=FORMAT_NAME_MEM,
        description='In-memory dataset I/O',
        ext='mem', modes={'r', 'w', 'a'}
    )


def _register_cli_commands(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard CLI commands.
    """

    cli_command_names = [
        'chunk',
        'compute',
        'benchmark',
        'dump',
        'edit',
        'extract',
        'gen',
        'genpts',
        'grid',
        'level',
        'optimize',
        'prune',
        'rectify',
        'resample',
        'serve',
        'vars2dim',
        'verify',
    ]

    for cli_command_name in cli_command_names:
        ext_registry.add_extension(
            loader=extension.import_component(f'xcube.cli.{cli_command_name}:{cli_command_name}'),
            point=EXTENSION_POINT_CLI_COMMANDS,
            name=cli_command_name
        )
