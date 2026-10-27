# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# https://opensource.org/licenses/MIT.

import unittest
from unittest.mock import Mock, patch

import fsspec
import geopandas as gpd

from xcube.core.store.fs.impl.geodataframe import (
    GeoDataFrameGeoJsonFsDataAccessor,
)


class GeoDataFrameGeoJsonFsDataAccessorTest(unittest.TestCase):
    def test_open_data_local(self):
        accessor = GeoDataFrameGeoJsonFsDataAccessor()
        data = Mock(spec=gpd.GeoDataFrame)
        fs = fsspec.filesystem("file")

        with patch(
            "xcube.core.store.fs.impl.geodataframe.gpd.read_file",
            return_value=data,
        ) as read_file:
            result = accessor.open_data(
                "sample.geojson",
                fs=fs,
                root=None,
                encoding="utf-8",
            )

        self.assertIs(data, result)
        read_file.assert_called_once_with(
            "sample.geojson",
            driver="GeoJSON",
            encoding="utf-8",
        )

    @patch("xcube.core.store.fs.impl.geodataframe.new_temp_file")
    @patch("xcube.core.store.fs.impl.geodataframe.gpd.read_file")
    def test_open_data_remote(self, read_file, new_temp_file):
        accessor = GeoDataFrameGeoJsonFsDataAccessor()
        data = Mock(spec=gpd.GeoDataFrame)
        fs = fsspec.filesystem("memory")
        new_temp_file.return_value = None, "sample.geojson"
        read_file.return_value = data

        with patch.object(fs, "get_file") as get_file:
            result = accessor.open_data(
                "sample.geojson",
                fs=fs,
                root="bucket",
                encoding="utf-8",
            )

        self.assertIs(data, result)
        get_file.assert_called_once_with(
            "sample.geojson", "sample.geojson"
        )
        read_file.assert_called_with(
            "sample.geojson",
            driver="GeoJSON",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
