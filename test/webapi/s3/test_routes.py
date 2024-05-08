# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from ..helpers import RoutesTestCase


class S3RoutesNewTest(RoutesTestCase):
    def test_fetch_head_s3_object(self):
        self._assert_fetch_s3_object(method="HEAD")

    def test_fetch_get_s3_object(self):
        self._assert_fetch_s3_object(method="GET")

    def _assert_fetch_s3_object(self, method):
        # response = self.fetch('/s3/datasets/demo.zarr', method=method)
        # self.assertResponseOK(response)
        # response = self.fetch('/s3/datasets/demo.zarr/', method=method)
        # self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/.zattrs", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/.zgroup", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/.zarray", method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch("/s3/datasets/demo.zarr/time/.zattrs", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/time/.zarray", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/time/.zgroup", method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch("/s3/datasets/demo.zarr/time/0", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/conc_chl/.zattrs", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/conc_chl/.zarray", method=method)
        self.assertResponseOK(response)
        response = self.fetch("/s3/datasets/demo.zarr/conc_chl/.zgroup", method=method)
        self.assertResourceNotFoundResponse(response)
        response = self.fetch("/s3/datasets/demo.zarr/conc_chl/3.2.4", method=method)
        self.assertResponseOK(response)
