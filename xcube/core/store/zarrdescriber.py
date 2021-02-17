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

from abc import abstractmethod
from abc import ABC
import json
import os
import s3fs
from typing import Sequence

from xcube.core.store import DatasetDescriptor

_DTYPE_TO_SAMPLE_TYPE = {
    '<i8': 'int8',
    '|u1': 'uint8',
    '<u2': 'uint16',
    '<u4': 'uint32',
    '<f4': 'float32',
    '<f8': 'float64'
}


class ZarrDescriber(ABC):
    """
    Base class for classes that derive DatasetDescriptors from a zarr directory without
    opening the data.
    """

    def describe(self, data_id: str) -> DatasetDescriptor:
        """
        Derives a DatasetDescriptor from a zarr directory

        :param data_id: The name of the zarr dataset.
        """
        # todo: check whether dataset is cube
        descriptor_dict = {'data_id': data_id}

        # Get general dataset information
        known_variable_names = None
        with self._open_file(data_id, '.zattrs') as zarr_attrs_file:
            zarr_attrs = json.load(zarr_attrs_file)
            self._set_if_included(zarr_attrs,
                                  descriptor_dict,
                                  ['time_coverage_duration'],
                                  'time_period')
            self._set_if_included(zarr_attrs,
                                  descriptor_dict,
                                  ['time_coverage_start', 'time_coverage_end'],
                                  'time_range')
            self._set_if_included(zarr_attrs,
                                  descriptor_dict,
                                  ['geospatial_lat_min', 'geospatial_lon_min',
                                   'geospatial_lat_max', 'geospatial_lon_max'],
                                  'bbox')
            if len(zarr_attrs.get('history', [])) > 0:
                cube_params = zarr_attrs.get('history', [])[0].get('cube_params', {})
                descriptor_dict['bbox'] = cube_params.get('bbox', None)
                known_variable_names = cube_params.get('variable_names', None)

        # Gather variable information and determine dimensions
        data_vars = {}
        dataset_dims = {}
        zarr_dir_content = self._list_zarr_dir_content(data_id)
        for content in zarr_dir_content:
            content = content.split('/')[-1]
            if content in ['.zattrs', '.zgroup']:
                continue
            # Using term variable now for clarity
            variable_name = content
            with self._open_file(data_id, f'{variable_name}/.zarray') as zarr_array_file:
                zarr_array = json.load(zarr_array_file)
                folder_dtype = zarr_array.get('dtype', None)
                if folder_dtype:
                    folder_dtype = _DTYPE_TO_SAMPLE_TYPE.get(folder_dtype, folder_dtype)
                folder_shape = zarr_array.get('shape', None)
            with self._open_file(data_id, f'{variable_name}/.zattrs') as zarr_attrs_file:
                zarr_attrs = json.load(zarr_attrs_file)
                var_dims = zarr_attrs.get('_ARRAY_DIMENSIONS', zarr_attrs.get('dimensions', None))
                if var_dims:
                    var_shape = zarr_attrs.get('shape', folder_shape)
                    if var_shape:
                        for i, var_dim in enumerate(var_dims):
                            if var_dim not in dataset_dims:
                                dataset_dims[var_dim] = var_shape[i]
                data_vars[variable_name] = dict(name=variable_name,
                                                dtype=zarr_attrs.get('data_type', folder_dtype),
                                                dims=var_dims,
                                                attrs=zarr_attrs)

        # Remove data_vars that are actually dimensions
        for dim in dataset_dims:
            if dim in data_vars:
                data_vars.pop(dim)
        to_be_removed = []
        for data_var in data_vars.keys():
            if data_var.endswith('bnds') or data_var.endswith('bounds'):
                to_be_removed.append(data_var)
                continue
            if known_variable_names is not None and data_var not in known_variable_names:
                to_be_removed.append(data_var)
        for remove in to_be_removed:
            data_vars.pop(remove)

        descriptor_dict['dims'] = dataset_dims
        descriptor_dict['data_vars'] = data_vars
        return DatasetDescriptor.from_dict(descriptor_dict)

    @staticmethod
    def _set_if_included(source: dict,
                         target: dict,
                         source_attrs: Sequence[str],
                         target_attr: str):
        for a in source_attrs:
            if a not in source:
                return
        if len(source_attrs) == 1:
            target[target_attr] = source[source_attrs[0]]
        else:
            target[target_attr] = tuple([source[attr_name] for attr_name in source_attrs])

    @abstractmethod
    def _open_file(self, data_id: str, file: str):
        pass

    @abstractmethod
    def _list_zarr_dir_content(self, data_id: str):
        pass


class S3ZarrDescriber(ZarrDescriber):

    def __init__(self, s3: s3fs.S3FileSystem, bucket_name: str):
        self._s3 = s3
        self._bucket_name = bucket_name

    def _open_file(self, data_id: str, file: str):
        return self._s3.open(f'{self._bucket_name}/{data_id}/{file}', 'r')

    def _list_zarr_dir_content(self, data_id: str):
        return self._s3.ls(f'{self._bucket_name}/{data_id}')


class DirectoryZarrDescriber(ZarrDescriber):

    def __init__(self, path: str):
        self._path = path

    def _open_file(self, data_id: str, file: str):
        return open(f'{self._path}/{data_id}/{file}', 'r')

    def _list_zarr_dir_content(self, data_id: str):
        return os.listdir(f'{self._path}/{data_id}')
