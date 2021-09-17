import unittest
from xcube.util.versions import get_xcube_versions
import xcube.version as xcube_version


class GetVersionsTest(unittest.TestCase):

    def test_get_xcube_versions(self):
        versions = get_xcube_versions()
        self.assertIsNotNone(versions)
        self.assertTrue('xarray' in versions)
        self.assertTrue('dask' in versions)
        self.assertTrue('zarr' in versions)
        self.assertTrue('tornado' in versions)
        self.assertTrue('pandas' in versions)
        self.assertTrue('numpy' in versions)
        self.assertTrue('geopandas' in versions)
        self.assertTrue('xcube' in versions)
        self.assertEqual(xcube_version.version, versions['xcube'])
