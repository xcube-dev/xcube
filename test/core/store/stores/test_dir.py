import os.path
import unittest

from xcube.core.store.stores.mem import DirectoryCubeStore


class DirectoryCubeStoreTest(unittest.TestCase):

    def test_iter_cubes(self):
        cube_store = DirectoryCubeStore(
            base_dir=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'serve', 'demo'),
            read_only=True)
        self.assertEqual({'cube-1-250-250', 'cube-5-100-200'},
                         set(cube_des.id for cube_des in cube_store.iter_cubes()))
