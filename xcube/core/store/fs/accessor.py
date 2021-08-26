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

import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import fsspec

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema
from ..accessor import DataOpener
from ..accessor import DataWriter
from ..error import DataStoreError

COMMON_FS_PARAMS_SCHEMA_PROPERTIES = dict(
    # passed to ``DirCache``, if the implementation supports
    # directory listing caching. Pass use_listings_cache=False
    # to disable such caching.
    use_listings_cache=JsonBooleanSchema(),
    listings_expiry_time=JsonNumberSchema(),
    max_paths=JsonIntegerSchema(),
    # If this is a cachable implementation, pass True here to force
    # creating a new instance even if a matching instance exists, and prevent
    # storing this instance.
    skip_instance_cache=JsonBooleanSchema(),
    asynchronous=JsonBooleanSchema(),
)

FS_PROTOCOL_PARAM_NAME = 'fs_protocol'
FS_PARAMS_PARAM_NAME = 'fs_params'
FS_PARAM_NAME = 'fs'


class FsAccessor:
    """
    Base class for accessing some filesystem.
    """

    @classmethod
    def get_fs_protocol(cls) -> str:
        """Get the filesystem protocol."""
        return 'abstract'

    @classmethod
    def get_fs_params_schema(cls) -> JsonObjectSchema:
        """Get the JSON schema of the filesystem parameters."""
        return JsonObjectSchema(
            properties=COMMON_FS_PARAMS_SCHEMA_PROPERTIES,
            additional_properties=True,
        )

    @classmethod
    def load_fs(cls, params: Dict[str, Any]) \
            -> Tuple[fsspec.AbstractFileSystem, Dict[str, Any]]:
        """
        Load a filesystem instance from *params*.

        Pops useful parameters from a copy of *params* to get
        or instantiate the filesystem.
        Returns the filesystem and *params* reduced by
        the used parameters.

        :param params: Parameters passed to a filesystem
            data store, opener, or writer call.
        :return: A tuple comprising the filesystem
            and the modified *params*.
        """
        params = dict(params)

        # Filesystem data stores pass "fs" kwarg to
        # data opener and writer calls.
        fs = params.pop(FS_PARAM_NAME, None)
        if fs is not None:
            assert_instance(fs, fsspec.AbstractFileSystem, name=FS_PARAM_NAME)
            return fs, params

        fs_protocol = cls.get_fs_protocol()
        if fs_protocol == 'abstract':
            fs_protocol = params.pop(FS_PROTOCOL_PARAM_NAME, None)
            if fs_protocol is None:
                raise DataStoreError(f"Cannot determine filesystem,"
                                     f" try using parameter"
                                     f" {FS_PROTOCOL_PARAM_NAME!r}")

        fs_params = params.pop(FS_PARAMS_PARAM_NAME, None)
        if fs_params is not None:
            assert_instance(fs_params, dict, name=FS_PARAMS_PARAM_NAME)
        try:
            return fsspec.filesystem(fs_protocol,
                                     **(fs_params or {})), params
        except (ValueError, ImportError):
            raise DataStoreError(f"Cannot instantiate"
                                 f" filesystem {fs_protocol!r}")

    @classmethod
    def add_fs_params_to_params_schema(
            cls, params_schema: JsonObjectSchema
    ) -> JsonObjectSchema:
        """
        Utility method to be used by subclasses to add the schema
        for the parameter "fs_param" to given *param_schema*.
        """
        params_schema = copy.deepcopy(params_schema)
        params_schema.properties[FS_PARAMS_PARAM_NAME] = \
            cls.get_fs_params_schema()
        return params_schema

    @classmethod
    def remove_fs_params_from_params_schema(
            cls, params_schema: JsonObjectSchema
    ) -> JsonObjectSchema:
        """
        Utility method to be used by subclasses to remove the schema
        for the parameter "fs_param" from given *param_schema*.
        """
        if FS_PARAMS_PARAM_NAME in params_schema.properties:
            params_schema = copy.deepcopy(params_schema)
            del params_schema.properties[FS_PARAMS_PARAM_NAME]
        return params_schema


class FsDataAccessor(DataOpener,
                     DataWriter,
                     FsAccessor,
                     ABC):
    """
    Abstract base class for data accessors that
    use an underlying filesystem.
    """

    @classmethod
    @abstractmethod
    def get_type_specifier(cls) -> str:
        """
        Get the data type specifier,
        for example "dataset" or "geodataframe".
        """

    @classmethod
    @abstractmethod
    def get_format_id(cls) -> str:
        """
        Get the format identifier,
        for example "zarr" or "geojson".
        """

    def get_delete_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                recursive=JsonBooleanSchema(),
                maxdepth=JsonIntegerSchema(),
                fs_params=self.get_fs_params_schema(),
            ),
            additional_properties=False,
        )

    def delete_data(self,
                    data_id: str,
                    **delete_params):
        fs, delete_params = self.load_fs(delete_params)
        fs.delete(data_id, **delete_params)
