# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json

from ..helpers import RoutesTestCase


class StatisticsRoutesTest(RoutesTestCase):

    def get_config_filename(self) -> str:
        """Get configuration filename.
        Default impl. returns ``'config.yml'``."""
        return "config-stats.yml"

    def test_fetch_post_statistics_ok(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-30+10:46:34",
            method="POST",
            body='{"type": "Point", "coordinates": [1.262, 50.243]}',
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert parsed_data["result"]["count"] == 1
        assert round(parsed_data["result"]["minimum"], 3) == 9.173
        assert round(parsed_data["result"]["maximum"], 3) == 9.173
        assert round(parsed_data["result"]["mean"], 3) == 9.173
        assert round(parsed_data["result"]["deviation"], 3) == 0.0

        response = self.fetch(
            "/statistics/cog_local/band_1",
            method="POST",
            body='{"type": "Point", "coordinates": [-105.810, 35.771]}',
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert parsed_data["result"]["count"] == 1
        assert round(parsed_data["result"]["minimum"], 3) == 102.0
        assert round(parsed_data["result"]["maximum"], 3) == 102.0
        assert round(parsed_data["result"]["mean"], 3) == 102.0
        assert round(parsed_data["result"]["deviation"], 3) == 0.0

    def test_fetch_post_statistics_missing_time_with_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/demo/conc_chl",
            method="POST",
            body='{"type": "Point", "coordinates": [1.768, 51.465]}',
        )
        self.assertBadRequestResponse(response, "Missing " "query parameter 'time'")

    def test_fetch_post_statistics_missing_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band_1",
            method="POST",
            body='{"type": "Point", "coordinates": [-105.591, 35.751]}',
        )
        self.assertResponseOK(response)

    def test_fetch_post_statistics_with_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band_1?time=2017-01-16+10:09:21",
            method="POST",
            body='{"type": "Point", "coordinates": [-105.591, 35.751]}',
        )
        self.assertBadRequestResponse(
            response,
            "Query parameter 'time' must not be given since "
            "dataset does not contain a 'time' dimension",
        )

    def test_fetch_post_statistics_invalid_geometry(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-16+10:09:21",
            method="POST",
            body="[1.768, 51.465, 11.542]",
        )
        self.assertBadRequestResponse(
            response, "Invalid " "GeoJSON geometry encountered"
        )
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-16+10:09:21",
            method="POST",
            body="[1.768]",
        )
        self.assertBadRequestResponse(
            response, "Invalid " "GeoJSON geometry encountered"
        )

    def test_crs_conversion_post_statistics_with_coordinates_outside_bounds(self):
        response = self.fetch(
            "/statistics/cog_local/band_1",
            method="POST",
            body='{"type": "Point", "coordinates": [-125.810, 35.771]}',
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert parsed_data["result"]["count"] == 0

    def test_crs_conversion_post_statistics_with_coordinates_inside_bounds(self):
        response = self.fetch(
            "/statistics/cog_local/band_1",
            method="POST",
            body='{"type": "Point", "coordinates": [-105.810, 35.171]}',
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert parsed_data["result"]["count"] == 1
        assert round(parsed_data["result"]["minimum"], 3) == 220.0
        assert round(parsed_data["result"]["maximum"], 3) == 220.0
        assert round(parsed_data["result"]["mean"], 3) == 220.0
        assert round(parsed_data["result"]["deviation"], 3) == 0.0

    def test_fetch_get_statistics_missing_time_with_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?lon=1.786&lat=51.465", method="GET"
        )
        self.assertBadRequestResponse(response, "Missing " "query parameter 'time'")

    def test_fetch_get_statistics_missing_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band_1?lon=-105.591&" "lat=35.751&type=Point",
            method="GET",
        )
        self.assertResponseOK(response)

    def test_fetch_get_statistics_with_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band_1?lon=-105.591&lat=35.751&"
            "type=Point&time=2017-01-16+10:09:21",
            method="GET",
        )
        self.assertBadRequestResponse(
            response,
            "Query parameter 'time' must not be given since "
            "dataset does not contain a 'time' dimension",
        )

    def test_fetch_get_statistics(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-30+10:46:34&"
            "lon=1.262&lat=50.243",
            method="GET",
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert round(parsed_data["result"]["value"], 3) == 9.173

        response = self.fetch(
            "/statistics/cog_local/band_1?lon=-105.810&lat=35.771&type=Point",
            method="GET",
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert round(parsed_data["result"]["value"], 3) == 102.0

    def test_crs_conversion_get_statistics_with_coordinates_outside_bounds(self):
        response = self.fetch(
            "/statistics/cog_local/band_1?lon=-125.810&lat=35.771&type=Point",
            method="GET",
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert parsed_data["result"] == {}

    def test_crs_conversion_get_statistics_with_coordinates_inside_bounds(self):
        response = self.fetch(
            "/statistics/cog_local/band_1?lon=-105.810&lat=35.171&type=Point",
            method="GET",
        )
        self.assertResponseOK(response)
        decoded_data = response.data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        assert round(parsed_data["result"]["value"], 3) == 220.0
