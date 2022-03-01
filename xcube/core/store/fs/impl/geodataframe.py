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

from abc import abstractmethod, ABC
from typing import Tuple, Optional

import geopandas as gpd
import pandas as pd

from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.temp import new_temp_file
from ..accessor import FsDataAccessor
from ..helpers import is_local_fs
from ...datatype import DataType
from ...datatype import GEO_DATA_FRAME_TYPE


class GeoDataFrameFsDataAccessor(FsDataAccessor, ABC):
    """
    Extension name: "geodataframe:<format_id>:<protocol>"
    """

    @classmethod
    def get_data_types(cls) -> Tuple[DataType, ...]:
        return GEO_DATA_FRAME_TYPE,

    @classmethod
    @abstractmethod
    def get_driver_name(cls) -> str:
        """Get the GeoDataFrame I/O driver name"""

    def get_open_data_params_schema(self, data_id: str = None) \
            -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                storage_options=self.get_storage_options_schema(),
                # TODO: add more, see https://geopandas.org/io.html
            ),
        )

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        # TODO: implement me correctly,
        #  this is not valid for shapefile AND geojson
        fs, root, open_params = self.load_fs(open_params)
        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file()
            fs.get_file(data_id, file_path)
        return gpd.read_file(file_path,
                             driver=self.get_driver_name(),
                             **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        return JsonObjectSchema(
            properties=dict(
                storage_options=self.get_storage_options_schema(),
                # TODO: add more, see https://geopandas.org/io.html
            ),
        )

    def write_data(self,
                   data: gpd.GeoDataFrame,
                   data_id: str,
                   **write_params) -> str:
        # TODO: implement me correctly,
        #  this is not valid for shapefile AND geojson
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        fs, root, write_params = self.load_fs(write_params)
        is_local = is_local_fs(fs)
        if is_local:
            file_path = data_id
        else:
            _, file_path = new_temp_file()
        data.to_file(file_path, driver=self.get_driver_name(), **write_params)
        if not is_local:
            fs.put_file(file_path, data_id)
        return data_id


class GeoDataFrameShapefileFsDataAccessor(GeoDataFrameFsDataAccessor, ABC):
    """
    Extension name: "geodataframe:shapefile:<protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'shapefile'

    @classmethod
    def get_driver_name(cls) -> str:
        return 'ESRI Shapefile'


class GeoDataFrameGeoJsonFsDataAccessor(GeoDataFrameFsDataAccessor, ABC):
    """
    Extension name: "geodataframe:geojson:<protocol>"
    """

    @classmethod
    def get_format_id(cls) -> str:
        return 'geojson'

    @classmethod
    def get_driver_name(cls) -> str:
        return 'GeoJSON'
