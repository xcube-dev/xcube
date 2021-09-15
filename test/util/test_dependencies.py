import unittest
from xcube.util.dependencies import get_xcube_dependencies
import xcube.version as xcube_version


class GetDependenciesTest(unittest.TestCase):

    def test_get_xcube_dependencies(self):
        dependencies = get_xcube_dependencies()
        self.assertIsNotNone(dependencies)
        self.assertTrue('xarray' in dependencies)
        self.assertTrue('dask' in dependencies)
        self.assertTrue('zarr' in dependencies)
        self.assertTrue('tornado' in dependencies)
        self.assertTrue('pandas' in dependencies)
        self.assertTrue('numpy' in dependencies)
        self.assertTrue('geopandas' in dependencies)
        self.assertTrue('xcube' in dependencies)
        self.assertEqual(xcube_version, dependencies['xcube'])
