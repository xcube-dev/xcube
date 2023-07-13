import os
from unittest import TestCase

import matplotlib.colors

from xcube.util.cmaps import Colormap
from xcube.util.cmaps import ColormapCategory
from xcube.util.cmaps import ColormapRegistry
from xcube.util.cmaps import DEFAULT_CMAP_NAME
from xcube.util.cmaps import ensure_cmaps_loaded
from xcube.util.cmaps import get_cmap
from xcube.util.cmaps import get_cmaps
from xcube.util.cmaps import load_snap_cpd_colormap

registry = ColormapRegistry()


class DeprecatedApiTest(TestCase):
    def test_get_cmap(self):
        cm_name, cmap = get_cmap("bone")
        self.assertEqual("bone", cm_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

        cm_name, cmap = get_cmap("bonex")
        self.assertEqual("viridis", cm_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

        cm_name, cmap = get_cmap("bone", num_colors=512)
        self.assertEqual("bone", cm_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_get_cmaps(self):
        cmaps = get_cmaps()
        self.assertIsInstance(cmaps, list)
        self.assertEqual(8, len(cmaps))
        self.assertIs(cmaps, get_cmaps())

    # noinspection PyMethodMayBeStatic
    def test_ensure_cmaps_loaded(self):
        ensure_cmaps_loaded()


class ColormapRegistryTest(TestCase):

    def setUp(self) -> None:
        self.registry = ColormapRegistry()

    def test_get_cmap(self):
        cmap_name, cmap = self.registry.get_cmap('plasma')
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

        cmap_name, cmap = self.registry.get_cmap('PLASMA')
        self.assertEqual(DEFAULT_CMAP_NAME, cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_get_cmap_alpha(self):
        cmap_name, cmap = self.registry.get_cmap('plasma_alpha')
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_get_cmap_reversed(self):
        cmap_name, cmap = self.registry.get_cmap('plasma_r')
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_get_cmap_reversed_alpha(self):
        cmap_name, cmap = self.registry.get_cmap('plasma_r_alpha')
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_get_cmap_num_colors(self):
        cmap_name, cmap = self.registry.get_cmap('plasma', num_colors=32)
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)

    def test_categories(self):
        categories = self.registry.categories
        self.assertIsInstance(categories, dict)
        self.assertGreaterEqual(len(categories), 8)
        self.assertIn('Perceptually Uniform Sequential', categories)
        self.assertIn('Sequential', categories)
        self.assertIn('Sequential (2)', categories)
        self.assertIn('Diverging', categories)
        self.assertIn('Qualitative', categories)
        self.assertIn('Cyclic', categories)
        self.assertIn('Ocean', categories)
        self.assertIn('Miscellaneous', categories)
        self.assertNotIn('Custom', categories)

    def test_category_descr(self):
        category = self.registry.categories.get(
            'Perceptually Uniform Sequential')
        self.assertEqual(
            'For many applications, a perceptually uniform colormap'
            ' is the best choice -'
            ' one in which equal steps in data are perceived as equal'
            ' steps in the color'
            ' space',
            category.desc
        )

    def test_colormaps(self):
        colormap = self.registry.colormaps.get("viridis")
        self.assertEqual('Perceptually Uniform Sequential',
                         colormap.cat_name)
        self.assertEqual('viridis', colormap.cm_name)
        self.assertEqual('iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAA'
                         'xWXB3AAAAxElEQVR4nI2TQZLEIAwDW+Jr84T5/1'
                         'PCHGzAkGRrDynZbclwIPro25FARg7FQhLYkBr9m'
                         'o/M9Fb/9K6vH32ddwnMHz3g5GKx9A2+dPirl91j'
                         '7pmpOvqqT36OM54z//XfGO/zVff3OXfv2w42Vnr'
                         'O+V6rME3NGYPlU9v0qAFP3vOZ9MlcPJVZHXP0ya'
                         'zrgQVvr31kGis76qYLc6guWuaWrh3hjf3Bw7d45'
                         'mqdO+rexrhLz7NHDS1/mwY0KWthKRQnN0Y0mR+s'
                         'DYkJKNzLygAAAABJRU5ErkJggg==',
                         colormap.cmap_png_base64)

    def test_to_json(self):
        obj = self.registry.to_json()
        self.assertIsInstance(obj, list)
        self.assertEqual(8, len(obj))
        for entry in obj:
            self.assertIsInstance(entry, list)
            self.assertEqual(3, len(entry))
            cat_name, cat_desc, cm_infos = entry
            self.assertIsInstance(cat_name, str)
            self.assertIsInstance(cat_desc, str)
            self.assertIsInstance(cm_infos, list)
            self.assertGreaterEqual(len(cm_infos), 3)
            for cm_info in cm_infos:
                self.assertIsInstance(cm_info, list)
                self.assertEqual(2, len(cm_info))
                cm_name, cm_png_base64 = cm_info
                self.assertIsInstance(cm_name, str)
                self.assertIsInstance(cm_png_base64, str)

    def test_category_ocean(self):
        ocean_category = self.registry.categories.get("Ocean")
        self.assertIsInstance(ocean_category, ColormapCategory)
        self.assertIn("thermal", ocean_category.cm_names)

    def test_colormaps_ocean(self):
        colormap = self.registry.colormaps.get("thermal")
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("Ocean", colormap.cat_name)
        self.assertEqual("thermal", colormap.cm_name)
        self.assertEqual('iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAA'
                         'xWXB3AAAAz0lEQVR4nHWSyZUEIQxDv0woHULnn1'
                         'p5DiWDgekDz7IkxKrx+aYIUA0BgapvmhiT06zDW'
                         'nED2SvrW2UQzRMMgh3HhQ+UTlfgXXhAiKsKCCUB'
                         'SGktkUp/OR0eTY+r++6vbLV15LV/6rDnXf62Bn1'
                         'fbd9n/jXvrOVrZ7xqOyfZcnvfc08+fWeVdXLPxX'
                         'V+9no2Xdr1u09QongWVqJYmHjPRucjfcENz4rn4'
                         'cs2Fnv9yb2PlPUwUxfZdELNs3CuRzP2vBDpj/sf'
                         'nzNneXL2TFz8H4K3dTwq9eoAAAAAAElFTkSuQmC'
                         'C',
                         colormap.cmap_png_base64)

    def test_load_snap_cpd_colormap(self):
        cmap_name = os.path.join(os.path.dirname(__file__),
                                 'chl_DeM2_200.cpd')
        colormap = load_snap_cpd_colormap(cmap_name)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("chl_DeM2_200", colormap.cm_name)
        self.assertIsInstance(
            colormap.cmap,
            matplotlib.colors.LinearSegmentedColormap,
        )
        self.assertIsInstance(
            colormap.norm,
            matplotlib.colors.Normalize,
        )

    def test_load_snap_cpd_colormap_invalid(self):
        cmap_name = os.path.join(os.path.dirname(__file__),
                                 'chl_DeM2_200_invalid_for_testing.cpd')
        with self.assertRaises(ValueError):
            load_snap_cpd_colormap(cmap_name)

    def test_load_snap_cpd_colormap_missing(self):
        cmap_name = 'test/webapi/im/chl_DeM2_200_not_existing.cpd'
        with self.assertRaises(FileNotFoundError):
            load_snap_cpd_colormap(cmap_name)


class ColormapTest(TestCase):

    def setUp(self) -> None:
        self.colormap = Colormap("coolwarm", cat_name="Diverging")

    def test_names(self):
        self.assertEqual("Diverging", self.colormap.cat_name)
        self.assertEqual("coolwarm", self.colormap.cm_name)

    def test_cmap(self):
        cmap = self.colormap.cmap
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)
        self.assertIs(cmap, self.colormap.cmap)
        self.assertEqual("coolwarm", cmap.name)

    def test_cmap_alpha(self):
        cmap = self.colormap.cmap_alpha
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)
        self.assertIs(cmap, self.colormap.cmap_alpha)
        self.assertEqual("coolwarm_alpha", cmap.name)

    def test_cmap_reversed(self):
        cmap = self.colormap.cmap_reversed
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)
        self.assertIs(cmap, self.colormap.cmap_reversed)
        self.assertEqual("coolwarm_r", cmap.name)

    def test_cmap_reversed_alpha(self):
        cmap = self.colormap.cmap_reversed_alpha
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)
        self.assertIs(cmap, self.colormap.cmap_reversed_alpha)
        self.assertEqual("coolwarm_r_alpha", cmap.name)

    def test_cmap_png_base64(self):
        base64 = self.colormap.cmap_png_base64
        self.assertIsInstance(base64, str)
        self.assertIs(base64, self.colormap.cmap_png_base64)
        self.assertEqual(
            'iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAAxWXB3AAAA'
            'u0lEQVR4nJWMwQEDIAwCScbqDB26e7VCHxpN1D76gsARezxf'
            'MjOYO9y7xm3j9nxfWYMnb27Hxuc29/EfV255LN5s45F2GL9R'
            'd4bJHd5QuJqhZpuvmbZep0fOB490I35osdD1nrt8X7yHB88+'
            'cg0v/tDuEZkqAyUFS2YSkLSw4pFBhDHYlYHxZ/m9E8/s6Eho'
            'KKSeT6b2U6m5FysXe01OkPYsdX94tpr3e2RNYPCNvWuJawJJ'
            '6C3wo7u+hS8PX1PsPRnqDgAAAABJRU5ErkJggg==',
            base64
        )
