import unittest

import xcube.version as xcube_version
from xcube.util.versions import get_xcube_versions


class GetVersionsTest(unittest.TestCase):

    def test_it_contains_what_is_expected(self):
        versions = get_xcube_versions()
        self.assertIsInstance(versions, dict)
        self.assertTrue(all(isinstance(k, str) for k in versions.keys()))
        self.assertTrue(all(isinstance(v, str) for v in versions.values()))
        self.assertIn('xarray', versions)
        self.assertIn('dask', versions)
        self.assertIn('zarr', versions)
        self.assertIn('tornado', versions)
        self.assertIn('pandas', versions)
        self.assertIn('numpy', versions)
        self.assertIn('geopandas', versions)
        self.assertIn('xcube', versions)
        self.assertEqual(xcube_version, versions['xcube'])

    def test_it_is_cached(self):
        versions = get_xcube_versions()
        self.assertIs(versions, get_xcube_versions())
