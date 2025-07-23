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

import fsspec
import rasterio
import rasterio.session
from rasterio.session import AWSSession
import s3fs
import rioxarray
import xarray as xr
from xcube.core.mldataset import LazyMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_instance, assert_true
from xcube.util.jsonencoder import to_json_value
from xcube.util.jsonschema import (
    JsonArraySchema,
    JsonBooleanSchema,
    JsonIntegerSchema,
    JsonNumberSchema,
    JsonObjectSchema
)

from .dataset import DatasetFsDataAccessor
from ...datatype import MULTI_LEVEL_DATASET_TYPE, DataType

RASTERIO_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=256, default=512),
                JsonNumberSchema(minimum=256, default=512),
            ),
            default=[512, 512],
        ),
        overview_level=JsonIntegerSchema(
            default=None,
            nullable=True,
            description="JPEG 2000 overview level. 0 is the first overview.",
        ),
        band_as_variable=JsonBooleanSchema(default=True)
    ),
    additional_properties=False,
)

MULTI_LEVEL_RASTERIO_OPEN_DATA_PARAMS_SCHEMA = JsonObjectSchema(
    properties=dict(
        tile_size=JsonArraySchema(
            items=(
                JsonNumberSchema(minimum=256, default=512),
                JsonNumberSchema(minimum=256, default=512),
            ),
            default=[512, 512],
        ),
        band_as_variable=JsonBooleanSchema(default=True)
    ),
    additional_properties=False,
)

class RasterioMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset for accessing files using rasterio.

    Args:
        data_id: data identifier
        items: list of items to be stacked
        open_params: opening parameters of odc.stack.load
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
        self._file_url = None

    def _get_overview_count(self):
        with rasterio.open(self._file_url) as rio_dataset:
            overviews = rio_dataset.overviews(1)
        return overviews

    def _get_num_levels_lazily(self) -> int:
        self._file_url = self._get_file_url()
        if isinstance(self._fs, fsspec.AbstractFileSystem):
            with DatasetJp2FsDataAccessor.create_env_session(self._fs) as env:
                env.__enter__()
                overviews = self._get_overview_count()
        else:
            assert_true(self._fs is None, message="invalid type for fs")
        return len(overviews) + 1

    def _get_dataset_lazily(self, index: int, parameters) -> xr.Dataset:
        tile_size = self._open_params.get("tile_size", (512, 512))
        self._file_url = self._get_file_url()
        return DatasetJp2FsDataAccessor.open_dataset(
            self._fs,
            self._file_url,
            tile_size,
            overview_level=index - 1 if index > 0 else None,
            band_as_variable=self._open_params.get("band_as_variable", True)
        )

    def _get_file_url(self):
        if isinstance(self._fs.protocol, str):
            protocol = self._fs.protocol
        else:
            protocol = self._fs.protocol[0]
        if self._root:
            return protocol + "://" + self._root + self._fs.sep + self._path
        return protocol + "://" + self._path


class DatasetRasterIoFsDataAccessor(DatasetFsDataAccessor, ABC):

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return RASTERIO_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> xr.Dataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)

        if isinstance(fs.protocol, str):
            protocol = fs.protocol
        else:
            protocol = fs.protocol[0]
        if root is not None:
            file_path = protocol + "://" + root + fs.sep +  data_id
        else:
            file_path = protocol + "://" + data_id
        tile_size = open_params.get("tile_size", (512, 512))
        overview_level = open_params.get("overview_level", None)
        band_as_variable = open_params.get("band_as_variable", True)
        return self.open_dataset(
            fs, file_path, tile_size,
            overview_level=overview_level,
            band_as_variable=band_as_variable
        )

    @classmethod
    def create_env_session(cls, fs):
        if isinstance(fs, s3fs.S3FileSystem):
            aws_unsigned = bool(fs.anon)
            aws_session = AWSSession(
                aws_unsigned=aws_unsigned,
                aws_secret_access_key=fs.secret,
                aws_access_key_id=fs.key,
                aws_session_token=fs.token,
                region_name=fs.client_kwargs.get("region_name", "eu-central-1"),
            )
            return rasterio.env.Env(
                session=aws_session, aws_no_sign_request=aws_unsigned, AWS_VIRTUAL_HOSTING=False
            )
        return rasterio.env.NullContextManager()

    @classmethod
    def open_dataset_with_rioxarray(
        cls, file_path, overview_level, tile_size,
        band_as_variable
    ) -> rioxarray.raster_array:
        return rioxarray.open_rasterio(
            file_path,
            overview_level=overview_level,
            chunks=dict(zip(("x", "y"), tile_size)),
            band_as_variable=band_as_variable
        )

    @classmethod
    def open_dataset(
        cls,
        fs,
        file_path: str,
        tile_size: tuple[int, int],
        *,
        overview_level: Optional[int] = None,
        band_as_variable: Optional[bool] = None,
    ) -> xr.Dataset:
        """
        A method to open a dataset using rioxarray, returns xarray.Dataset
        @param fs: abstract file system
        @type fs: fsspec.AbstractFileSystem object.
        @param file_path: path to the file
        @type file_path: str
        @param overview_level: the overview level of GeoTIFF, 0 is the first
               overview and None means full resolution.
        @type overview_level: int
        @param tile_size: tile size as tuple.
        @type tile_size: tuple
        @param band_as_variable: If True, will load bands in a raster
            to separate variables.
        @type bands_as_variable
        """

        if isinstance(fs, fsspec.AbstractFileSystem):
            with cls.create_env_session(fs):
                dataset = cls.open_dataset_with_rioxarray(
                    file_path, overview_level, tile_size,
                    band_as_variable
                )
        else:
            assert_true(fs is None, message="invalid type for fs")
        arrays = {}
        if isinstance(dataset, xr.DataArray):
            if dataset.ndim != 2:
                raise RuntimeError(f"Invalid number of dimensions, was {dataset.ndim}")
            name = f"{dataset.name or 'band'}"
            arrays[name] = dataset
            dataset = xr.Dataset(arrays, attrs=dict(source=file_path))
        if "spatial_ref" in dataset.coords:
            for data_var in dataset.data_vars.values():
                data_var.attrs["grid_mapping"] = "spatial_ref"
        # rioxarray may return non-JSON-serializable metadata
        # attribute values.
        # We have seen _FillValue of type np.uint8
        cls._sanitize_dataset_attrs(dataset)

        return dataset

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        raise NotImplementedError("Writing not yet supported")

    def write_data(
        self, data: xr.Dataset, data_id: str, replace=False, **write_params
    ) -> str:
        raise NotImplementedError("Writing not yet supported")

    @classmethod
    def _sanitize_dataset_attrs(cls, dataset):
        dataset.attrs.update(to_json_value(dataset.attrs))
        for var in dataset.variables.values():
            var.attrs.update(to_json_value(var.attrs))


class DatasetJp2FsDataAccessor(DatasetRasterIoFsDataAccessor):
    """
    Opener/writer extension name: 'dataset:jpeg2000:<protocol>'.
    """

    @classmethod
    def get_format_id(cls) -> str:
        return "jpeg2000"


class DatasetGeoTiffFsDataAccessor(DatasetRasterIoFsDataAccessor):
    """
    Opener/writer extension name: 'dataset:geotiff:<protocol>'.
    """

    @classmethod
    def get_format_id(cls) -> str:
        return "geotiff"


# noinspection PyAbstractClass
class MultiLevelDatasetRasterioFsDataAccessor(DatasetFsDataAccessor):

    @classmethod
    def get_data_type(cls) -> DataType:
        return MULTI_LEVEL_DATASET_TYPE

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        return MULTI_LEVEL_RASTERIO_OPEN_DATA_PARAMS_SCHEMA

    def open_data(self, data_id: str, **open_params) -> MultiLevelDataset:
        assert_instance(data_id, str, name="data_id")
        fs, root, open_params = self.load_fs(open_params)
        return RasterioMultiLevelDataset(fs, root, data_id, **open_params)


# noinspection PyAbstractClass
class MultiLevelDatasetJp2FsDataAccessor(
    MultiLevelDatasetRasterioFsDataAccessor, DatasetJp2FsDataAccessor
):
    """
    Opener/writer extension name: "mldataset:jpeg2000:<protocol>"
    """

# noinspection PyAbstractClass
class MultiLevelDatasetGeoTiffFsDataAccessor(
    MultiLevelDatasetRasterioFsDataAccessor, DatasetGeoTiffFsDataAccessor
):
    """
    Opener/writer extension name: "mldataset:geotiff:<protocol>"
    """
