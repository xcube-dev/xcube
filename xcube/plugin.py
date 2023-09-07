# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
from xcube.constants import EXTENSION_POINT_DATA_OPENERS
from xcube.constants import EXTENSION_POINT_DATA_STORES
from xcube.constants import EXTENSION_POINT_DATA_WRITERS
from xcube.constants import EXTENSION_POINT_INPUT_PROCESSORS
from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.constants import EXTENSION_POINT_SERVER_FRAMEWORKS
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
    _register_data_stores(ext_registry)
    _register_data_accessors(ext_registry)
    _register_server_apis(ext_registry)
    _register_server_frameworks(ext_registry)


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


_FS_STORAGE_ITEMS = (
    ('file', 'local filesystem'),
    ('s3', 'AWS S3 compatible object storage'),
    ('abfs', 'Azure blob compatible object storage'),
    ('memory', 'in-memory filesystem'),
    ('ftp', 'FTP filesystem')
)

_FS_DATA_ACCESSOR_ITEMS = (
    ('dataset', 'netcdf',
     'xarray.Dataset in NetCDF format'),
    ('dataset', 'zarr',
     'xarray.Dataset in Zarr format'),
    ('dataset', 'levels',
     'xarray.Dataset in leveled Zarr format'),
    ('mldataset', 'levels',
     'xcube.core.mldataset.MultiLevelDataset in leveled Zarr format'),
    ('dataset', 'geotiff',
     'xarray.Dataset in GeoTIFF or COG format'),
    ('mldataset', 'geotiff',
     'xcube.core.mldataset.MultiLevelDataset in GeoTIFF or COG format'),
    ('geodataframe', 'shapefile',
     'gpd.GeoDataFrame in ESRI Shapefile format'),
    ('geodataframe', 'geojson',
     'gpd.GeoDataFrame in GeoJSON format'),
)

_FS_DATA_OPENER_ITEMS = _FS_DATA_ACCESSOR_ITEMS
_FS_DATA_WRITER_ITEMS = _FS_DATA_ACCESSOR_ITEMS


def _register_data_stores(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard data stores.
    """
    factory = 'xcube.core.store.fs.registry:get_fs_data_store_class'

    for storage_id, storage_description in _FS_STORAGE_ITEMS:
        loader = extension.import_component(factory, call_args=[storage_id])
        ext_registry.add_extension(
            point=EXTENSION_POINT_DATA_STORES,
            loader=loader,
            name=storage_id,
            description=f'Data store that uses a {storage_description}'
        )


def _register_data_accessors(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard data accessors.
    """
    factory = 'xcube.core.store.fs.registry:get_fs_data_accessor_class'

    # noinspection PyShadowingNames
    def _add_fs_data_accessor_ext(point: str,
                                  ext_type: str,
                                  protocol: str,
                                  data_type: str,
                                  format_id: str):
        factory_args = (protocol, data_type, format_id)
        loader = extension.import_component(factory,
                                            call_args=factory_args)
        ext_registry.add_extension(
            point=point,
            loader=loader,
            name=f'{data_type}:{format_id}:{protocol}',
            description=f'Data {ext_type} for'
                        f' a {data_accessor_description}'
                        f' in {storage_description}'
        )

    for protocol, storage_description in _FS_STORAGE_ITEMS:
        for data_type, format_id, data_accessor_description \
                in _FS_DATA_OPENER_ITEMS:
            _add_fs_data_accessor_ext(EXTENSION_POINT_DATA_OPENERS,
                                      'opener',
                                      protocol,
                                      data_type,
                                      format_id)
        for data_type, format_id, data_accessor_description \
                in _FS_DATA_WRITER_ITEMS:
            _add_fs_data_accessor_ext(EXTENSION_POINT_DATA_WRITERS,
                                      'writer',
                                      protocol,
                                      data_type,
                                      format_id)


def _register_cli_commands(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard CLI commands.
    """

    cli_command_names = [
        'chunk',
        'compute',
        'benchmark',
        'dump',
        'extract',
        'gen',
        'gen2',
        'genpts',
        'grid',
        'level',
        'optimize',
        'patch',
        'prune',
        'rectify',
        'resample',
        'serve',
        'vars2dim',
        'verify',
        'versions',

        # Experimental + Hidden
        'io',
    ]

    for cli_command_name in cli_command_names:
        ext_registry.add_extension(
            loader=extension.import_component(f'xcube.cli.{cli_command_name}:{cli_command_name}'),
            point=EXTENSION_POINT_CLI_COMMANDS,
            name=cli_command_name
        )


def _register_server_apis(ext_registry: extension.ExtensionRegistry):
    """
    Register xcube's standard server APIs.
    """
    server_api_names = [
        'meta',
        'auth',
        'compute',
        'places',
        'styles',
        'datasets',
        'tiles',
        'timeseries',
        'volumes',
        'ows.coverages',
        'ows.stac',
        'ows.wmts',
        's3',
        'viewer',
    ]
    for api_name in server_api_names:
        ext_registry.add_extension(
            loader=extension.import_component(
                f'xcube.webapi.{api_name}:api'
            ),
            point=EXTENSION_POINT_SERVER_APIS,
            name=api_name
        )


def _register_server_frameworks(ext_registry: extension.ExtensionRegistry):
    server_framework_names = [
        'tornado',
        'flask',
    ]
    for framework_name in server_framework_names:
        ext_registry.add_extension(
            loader=extension.import_component(
                    f'xcube.server.webservers.{framework_name}'
                    f':{framework_name.capitalize()}Framework',
                ),
            point=EXTENSION_POINT_SERVER_FRAMEWORKS,
            name=framework_name
        )
