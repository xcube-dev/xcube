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

import geopandas as gpd
import pandas as pd

from xcube.core.store.accessor import DataOpener
from xcube.core.store.accessor import DataWriter
from xcube.core.store.accessors.posix import PosixDataDeleterMixin
from xcube.util.assertions import assert_instance
from xcube.util.jsonschema import JsonObjectSchema


class GdfShapefilePosixAccessor(PosixDataDeleterMixin, DataWriter, DataOpener):
    """
    Extension name: "geodataframe:shapefile:posix"
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me, see https://geopandas.org/io.html
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        return gpd.read_file(data_id, driver='ESRI Shapefile', **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me, see https://geopandas.org/io.html
        return JsonObjectSchema()

    def write_data(self, data: gpd.GeoDataFrame, data_id: str, **write_params):
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        data.to_file(data_id, driver='ESRI Shapefile', **write_params)


class GdfGeoJsonPosixAccessor(PosixDataDeleterMixin, DataWriter, DataOpener):
    """
    Extension name: "geodataframe:geojson:posix"
    """

    def get_open_data_params_schema(self, data_id: str = None) -> JsonObjectSchema:
        # TODO: implement me, see https://geopandas.org/io.html
        return JsonObjectSchema()

    def open_data(self, data_id: str, **open_params) -> gpd.GeoDataFrame:
        return gpd.read_file(data_id, driver='GeoJSON', **open_params)

    def get_write_data_params_schema(self) -> JsonObjectSchema:
        # TODO: implement me, see https://geopandas.org/io.html
        return JsonObjectSchema()

    def write_data(self, data: gpd.GeoDataFrame, gdf_id: str, **write_params):
        assert_instance(data, (gpd.GeoDataFrame, pd.DataFrame), 'data')
        data.to_file(gdf_id, driver='GeoJSON', **write_params)
