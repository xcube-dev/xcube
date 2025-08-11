# The MIT License (MIT)
# Copyright (c) 2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from abc import ABC
from typing import Any, Optional

import dask
import fsspec
import rasterio
import rasterio.session
import rioxarray
import s3fs
import xarray as xr
from rasterio.session import AWSSession

from xcube.core.mldataset import LazyMultiLevelDataset, MultiLevelDataset
from xcube.util.assertions import assert_instance
from xcube.util.jsonencoder import to_json_value
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonIntegerSchema,
    JsonNumberSchema,
    JsonObjectSchema,
)

from ...datatype import DATASET_TYPE, MULTI_LEVEL_DATASET_TYPE, DataType
from ...error import DataStoreError
from ..accessor import FsDataAccessor

RASTERIO_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=512, default=1024),
                JsonNumberSchema(minimum=512, default=1024),
            ),
            default=[1024, 1024],
        ),
        overview_level=JsonIntegerSchema(
            default=None,
            nullable=True,
            description="JPEG 2000 overview level. 0 is the first overview.",
        ),
    ),
    additional_properties=False,
)

MULTI_LEVEL_RASTERIO_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=512, default=1024),
                JsonNumberSchema(minimum=512, default=1024),
            ),
            default=[1024, 1024],
        ),
    ),
    additional_properties=False,
)


class RasterIoAccessor:
    def __init__(self, fs):
        if isinstance(fs, s3fs.S3FileSystem):
            endpoint_url = fs.client_kwargs.get("endpoint_url")
            endpoint_url = endpoint_url.split("://", 2)[-1] if endpoint_url else None
            aws_unsigned = bool(fs.anon)
            aws_session = AWSSession(
                aws_unsigned=aws_unsigned,
                aws_secret_access_key=fs.secret,
                aws_access_key_id=fs.key,
                aws_session_token=fs.token,
                region_name=fs.client_kwargs.get("region_name", "eu-central-1"),
                endpoint_url=endpoint_url,
            )
            self.env = rasterio.env.Env(
                session=aws_session,
                aws_no_sign_request=aws_unsigned,
                AWS_VIRTUAL_HOSTING=False,
            )
            self.env = self.env.__enter__()
            return
        self.env = rasterio.env.NullContextManager()

    # noinspection PyMethodMayBeStatic
    def get_overview_count(self, file_url):
        with rasterio.open(file_url) as rio_dataset:
            overviews = rio_dataset.overviews(1)
        return overviews

    # noinspection PyMethodMayBeStatic
    def open_dataset_with_rioxarray(
        self, file_path, overview_level, tile_size
    ) -> rioxarray.raster_array:
        return rioxarray.open_rasterio(
            file_path,
            overview_level=overview_level,
            chunks=dict(zip(("x", "y"), tile_size)),
            band_as_variable=True,
        )

    def open_dataset(
        self,
        file_path: str,
        tile_size: tuple[int, int],
        *,
        overview_level: Optional[int] = None,
    ) -> xr.Dataset:
        """
        A method to open a dataset using rioxarray, returns xarray.Dataset.

        Args:
            file_path: path to the file
            tile_size: tile size as tuple.
            overview_level: the overview level of GeoTIFF,
                0 is the first overview and None means full resolution.

        Returns:
            The opened data as xarray.Dataset
        """
        dataset = self.open_dataset_with_rioxarray(file_path, overview_level, tile_size)
        if "spatial_ref" in dataset.coords:
            for data_var in dataset.data_vars.values():
                data_var.attrs["grid_mapping"] = "spatial_ref"
        # rioxarray may return non-JSON-serializable metadata
        # attribute values.
        # We have seen _FillValue of type np.uint8
        self._sanitize_dataset_attrs(dataset)

        return dataset

    @classmethod
    def _sanitize_dataset_attrs(cls, dataset):
        dataset.attrs.update(to_json_value(dataset.attrs))
        for var in dataset.variables.values():
            var.attrs.update(to_json_value(var.attrs))


class RasterioMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for accessing files using rasterio.

        fs: abstract file system
        root: An optional root pointing to where the data is located
        data_id: the data id
        open_params: Any additional parameters to be considered when opening the data
    """

    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        root: Optional[str],
        data_id: str,
        **open_params: dict[str, Any],
    ):
        super().__init__(ds_id=data_id)
        self._fs = fs
        self._root = root
        self._path = data_id
        self._open_params = open_params
        self._file_url = self._get_file_url()
        self._rio_accessor = RasterIoAccessor(self._fs)

    def _get_num_levels_lazily(self) -> int:
        overviews = self._rio_accessor.get_overview_count(self._file_url)
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        tile_size = self._open_params.get("tile_size", (1024, 1024))
        self._file_url = self._get_file_url()
        return self._rio_accessor.open_dataset(
            self._file_url, tile_size, overview_level=index - 1 if index > 0 else None
        )

    def _get_file_url(self):
        protocol = (
            self._fs.protocol
            if isinstance(self._fs.protocol, str)
            else self._fs.protocol[0]
        )
        if self._root:
            return protocol + "://" + self._root + self._fs.sep + self._path
        return protocol + "://" + self._path


class DatasetRasterIoFsDataAccessor(FsDataAccessor, ABC):
    def __init__(self):
        # required to keep accessors alive and therefore sessions open
        self._rio_accessors = {}

    @classmethod
    def get_data_type(cls) -> DataType:
        return DATASET_TYPE

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return RASTERIO_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)

        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        if root is not None:
            file_path = protocol + "://" + root + fs.sep + data_id
        else:
            file_path = protocol + "://" + data_id
        tile_size = open_params.get("tile_size", (1024, 1024))
        overview_level = open_params.get("overview_level", None)
        if fs not in self._rio_accessors.keys():
            self._rio_accessors[fs] = RasterIoAccessor(fs)
        rio_accessor = self._rio_accessors[fs]
        return rio_accessor.open_dataset(
            file_path, tile_size, overview_level=overview_level
        )

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        raise NotImplementedError("Writing not yet supported")

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        raise NotImplementedError("Writing not yet supported")


class DatasetJ2kFsDataAccessor(DatasetRasterIoFsDataAccessor):
    """
    Opener/writer extension name: 'dataset:j2k:<protocol>'.
    """

    def __init__(self):
        if dask.config.get("scheduler", "") != "single-threaded":
            raise DataStoreError(
                "For opening JPEG 2000 please set the scheduler in your "
                "dask configuration to 'single-threaded', e.g., by executing "
                "dask.config.set(scheduler='single-threaded')"
            )
        super().__init__()

    @classmethod
    def get_format_id(cls) -> str:
        return "j2k"


class DatasetGeoTiffFsDataAccessor(DatasetRasterIoFsDataAccessor):
    """
    Opener/writer extension name: 'dataset:geotiff:<protocol>'.
    """

    @classmethod
    def get_format_id(cls) -> str:
        return "geotiff"


class MultiLevelDatasetRasterioFsDataAccessor(FsDataAccessor, ABC):
    @classmethod
    def get_data_type(cls) -> DataType:
        return MULTI_LEVEL_DATASET_TYPE

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return MULTI_LEVEL_RASTERIO_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)
        return RasterioMultiLevelDataset(fs, root, data_id, **open_params)


class MultiLevelDatasetJ2kFsDataAccessor(
    MultiLevelDatasetRasterioFsDataAccessor, DatasetJ2kFsDataAccessor
):
    """
    Opener/writer extension name: "mldataset:j2k:<protocol>"
    """


class MultiLevelDatasetGeoTiffFsDataAccessor(
    MultiLevelDatasetRasterioFsDataAccessor, DatasetGeoTiffFsDataAccessor
):
    """
    Opener/writer extension name: "mldataset:geotiff:<protocol>"
    """
