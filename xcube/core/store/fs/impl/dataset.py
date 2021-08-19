# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from abc import ABC

import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.temp import new_temp_file
from ..accessor import FsDataAccessor
from ..helpers import is_local_fs
from ...error import DataStoreError


class DatasetFsDataAccessor(FsDataAccessor, ABC):

    @classmethod
    def get_type_specifier(cls) -> str:
        return 'dataset'


class DatasetZarrFsDataAccessor(DatasetFsDataAccessor, ABC):
    """
    Opener/writer extension name: "dataset:zarr:<fs_protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'zarr'

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_open_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                group=JsonStringSchema(
                    description='Group path.'
                                ' (a.k.a. path in zarr terminology.).',
                    min_length=1,
                ),
                chunks=JsonObjectSchema(
                    description='Optional chunk sizes along each dimension.'
                                ' Chunk size values may be None, "auto"'
                                ' or an integer value.',
                    examples=[{'time': None, 'lat': 'auto', 'lon': 90},
                              {'time': 1, 'y': 512, 'x': 512}],
                    additional_properties=True,
                ),
                decode_cf=JsonBooleanSchema(
                    description='Whether to decode these variables,'
                                ' assuming they were saved according to'
                                ' CF conventions.',
                    default=True,
                ),
                mask_and_scale=JsonBooleanSchema(
                    description='If True, replace array values equal'
                                ' to attribute "_FillValue" with NaN. '
                                ' Use "scaling_factor" and "add_offset"'
                                ' attributes to compute actual values.',
                    default=True,
                ),
                decode_times=JsonBooleanSchema(
                    description='If True, decode times encoded in the'
                                ' standard NetCDF datetime format '
                                'into datetime objects. Otherwise,'
                                ' leave them encoded as numbers.',
                    default=True,
                ),
                decode_coords=JsonBooleanSchema(
                    description='If True, decode the \"coordinates\"'
                                ' attribute to identify coordinates in '
                                'the resulting dataset.',
                    default=True,
                ),
                drop_variables=JsonArraySchema(
                    items=JsonStringSchema(min_length=1),
                ),
                consolidated=JsonBooleanSchema(
                    description='Whether to open the store using'
                                ' Zarr\'s consolidated metadata '
                                'capability. Only works for stores that'
                                ' have already been consolidated.',
                    default=False,
                ),
                fs_params=self.get_fs_params_schema(),
            ),
            required=[],
            additional_properties=False
        )

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        fs, open_params = self.load_fs(open_params)
        zarr_store = fs.get_mapper(data_id)
        consolidated = open_params.pop('consolidated',
                                       fs.exists(f'{data_id}/.zmetadata'))
        try:
            return xr.open_zarr(zarr_store,
                                consolidated=consolidated,
                                **open_params)
        except ValueError as e:
            raise DataStoreError(f'failed to open {data_id}: {e}') from e

    # noinspection PyMethodMayBeStatic
    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                fs_params=self.get_fs_params_schema(),
                group=JsonStringSchema(
                    description='Group path.'
                                ' (a.k.a. path in zarr terminology.).',
                    min_length=1,
                ),
                encoding=JsonObjectSchema(
                    description='Nested dictionary with variable'
                                ' names as keys and dictionaries of'
                                ' variable specific encodings as values.',
                    examples=[{
                        'my_variable': {
                            'dtype': 'int16',
                            'scale_factor': 0.1
                        }
                    }],
                    additional_properties=True,
                ),
                consolidated=JsonBooleanSchema(
                    description='If True, apply Zarr’s consolidate_metadata()'
                                ' function to the store after writing.'
                ),
                append_dim=JsonStringSchema(
                    description='If set, the dimension on which the'
                                ' data will be appended.',
                    min_length=1,
                ),
            ),
            additional_properties=False
        )

    def write_data(self,
                   data: xr.Dataset,
                   data_id: str,
                   replace=False,
                   **write_params):
        assert_instance(data, xr.Dataset, 'data')
        fs, write_params = self.load_fs(write_params)
        zarr_store = fs.get_mapper(data_id, create=True)
        consolidated = write_params.pop('consolidated', True)
        try:
            data.to_zarr(zarr_store,
                         mode='w' if replace else None,
                         consolidated=consolidated,
                         **write_params)
        except ValueError as e:
            raise DataStoreError(f'failed to write {data_id}: {e}') from e

    def delete_data(self,
                    data_id: str,
                    **delete_params):
        fs, delete_params = self.load_fs(delete_params)
        delete_params.pop('recursive', None)
        fs.delete(data_id, recursive=True, **delete_params)


class DatasetNetcdfFsDataAccessor(DatasetFsDataAccessor, ABC):
    """
    Opener/writer extension name: "dataset:netcdf:<fs_protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'netcdf'

    def get_open_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                fs_params=self.get_fs_params_schema()
                # TODO: add more from xr.open_dataset()
            ),
            additional_properties=True,
        )

    def open_data(self,
                  data_id: str,
                  **open_params) -> xr.Dataset:
        fs, open_params = self.load_fs(open_params)

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = open_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'rb') as file:
        #     return xr.open_dataset(file, engine=engine, **open_params)

        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file(suffix='.nc')
            fs.get_file(data_id, file_path)
        engine = open_params.pop('engine', 'netcdf4')
        return xr.open_dataset(file_path, engine=engine, **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                fs_params=self.get_fs_params_schema()
                # TODO: add more from ds.to_netcdf()
            ),
            additional_properties=True,
        )

    def write_data(self,
                   data: xr.Dataset,
                   data_id: str,
                   replace=False,
                   **write_params):
        fs, write_params = self.load_fs(write_params)
        assert_instance(data, xr.Dataset, 'data')
        if not replace and fs.exists(data_id):
            raise DataStoreError(f'Data resource {data_id} already exists')

        # This doesn't yet work as expected with fsspec and netcdf:
        # engine = write_params.pop('engine', 'scipy')
        # with fs.open(data_id, 'wb') as file:
        #     data.to_netcdf(file, engine=engine, **write_params)

        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file(suffix='.nc')
        engine = write_params.pop('engine', 'netcdf4')
        data.to_netcdf(file_path, engine=engine, **write_params)
        if not is_local:
            fs.put_file(file_path, data_id)