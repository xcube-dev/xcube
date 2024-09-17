# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class StatisticsRoutesTest(RoutesTestCase):

    def get_config_filename(self) -> str:
        """Get configuration filename.
        Default impl. returns ``'config.yml'``."""
        return "config-stats.yml"

    def test_fetch_post_statistics_ok(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-16+10:09:21",
            method="POST",
            body='{"type": "Point", "coordinates": [1.768, 51.465]}',
        )
        self.assertResponseOK(response)

    def test_fetch_post_statistics_missing_time_with_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/demo/conc_chl",
            method="POST",
            body='{"type": "Point", "coordinates": [1.768, 51.465]}',
        )
        self.assertBadRequestResponse(response, "Missing " "query parameter 'time'")

    def test_fetch_post_statistics_missing_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band-1",
            method="POST",
            body='{"type": "Point", "coordinates": [-105.591, 35.751]}',
        )
        self.assertResponseOK(response)

    def test_fetch_post_statistics_with_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band-1?time=2017-01-16+10:09:21",
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
            body="[1.768, 51.465]",
        )
        self.assertBadRequestResponse(
            response, "Invalid " "GeoJSON geometry encountered"
        )

    def test_fetch_get_statistics_ok(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?"
            "lat=1.786&lon=51.465&time=2017-01-16+10:09:21",
            method="GET",
        )
        self.assertResponseOK(response)

    def test_fetch_get_statistics_missing_time_with_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?lat=1.786&lon=51.465", method="GET"
        )
        self.assertBadRequestResponse(response, "Missing " "query parameter 'time'")

    def test_fetch_get_statistics_missing_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band-1?lat=-105.591&" "lon=35.751&type=Point",
            method="GET",
        )
        self.assertResponseOK(response)

    def test_fetch_get_statistics_with_time_without_time_dimension_dataset(self):
        response = self.fetch(
            "/statistics/cog_local/band-1?lat=-105.591&lon=35.751&"
            "type=Point&time=2017-01-16+10:09:21",
            method="GET",
            body='{"type": "Point", "coordinates": [-105.591, 35.751]}',
        )
        self.assertBadRequestResponse(
            response,
            "Query parameter 'time' must not be given since "
            "dataset does not contain a 'time' dimension",
        )

    def test_fetch_get_statistics_invalid_geometry(self):
        response = self.fetch(
            "/statistics/demo/conc_chl?time=2017-01-16+10:09:21&"
            "lon=1.768&lat=51.465",
            method="GET",
        )
        self.assertResponseOK(response)
