# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from abc import ABC
from typing import Optional, Tuple, Dict, Any

import fsspec
import rasterio
import xarray as xr

from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataType
from xcube.core.store import MULTI_LEVEL_DATASET_TYPE
from xcube.core.store.fs.impl.dataset import DatasetGeoTiffFsDataAccessor
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonNumberSchema
from xcube.util.jsonschema import JsonObjectSchema


class GeoTIFFMultiLevelDataset(LazyMultiLevelDataset):
    """
    A multi-level dataset for GeoTIFF format

    @param fs: fsspec.AbstractFileSystem object.
    @param root: Optional root path identifier.
    @param data_id: dataset identifier.
    @param open_params: keyword arguments.
    """

    def __init__(self,
                 fs: fsspec.AbstractFileSystem,
                 root: Optional[str],
                 data_id: str,
                 **open_params: Dict[str, Any]):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._path = data_id
        self._open_params = open_params
        self._file_url = None

    def _get_num_levels_lazily(self) -> int:
        self._file_url = self._get_file_url()
        if isinstance(self._fs.protocol, str):
            with rasterio.open(self._file_url) as rio_dataset:
                overviews = rio_dataset.overviews(1)
        else:
            if self._fs.secret is None or self._fs.key is None:
                AWS_NO_SIGN_REQUEST = True
            else:
                AWS_NO_SIGN_REQUEST = False
            Session = rasterio.env.Env(
                region_name=self._fs.kwargs.get('region_name', 'eu-central-1'),
                AWS_NO_SIGN_REQUEST=AWS_NO_SIGN_REQUEST,
                aws_session_token=self._fs.token,
                aws_access_key_id=self._fs.key,
                aws_secret_access_key=self._fs.secret

            )
            with Session:
                with rasterio.open(self._file_url) as rio_dataset: \
                        overviews = rio_dataset.overviews(1)
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) \
            -> xr.Dataset:
        tile_size = self._open_params.get("tile_size", (512, 512))
        self._file_url = self._get_file_url()
        return DatasetGeoTiffFsDataAccessor.open_dataset(
            self._fs,
            self._file_url,
            tile_size,
            overview_level=index - 1 if index > 0 else None
        )

    def _get_file_url(self):
        if isinstance(self._fs.protocol, str):
            protocol = self._fs.protocol
            url = protocol + "://" + self._path
        else:
            protocol = self._fs.protocol[0]
            url = protocol + "://" + self._path
        return url


MULTI_LEVEL_GEOTIFF_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=256, default=512),
                JsonNumberSchema(minimum=256, default=512)
            ),
            default=[512, 512]
        ),
    ),
    additional_properties=False,
)


class MultiLevelDatasetGeoTiffFsDataAccessor(DatasetGeoTiffFsDataAccessor, ABC):
    """
    Opener/writer extension name: "mldataset:geotiff:<protocol>"
    and "dataset:geotiff:<protocol>"
    """

    @classmethod
    def get_data_types(cls) -> Tuple[DataType, ...]:
        return MULTI_LEVEL_DATASET_TYPE, DATASET_TYPE

    def get_open_data_params_schema(self,
                                    data_id: str = None) -> JsonObjectSchema:
        return MULTI_LEVEL_GEOTIFF_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name='data_id')
        fs, root, open_params = self.load_fs(open_params)
        return GeoTIFFMultiLevelDataset(fs, root, data_id, **open_params)
