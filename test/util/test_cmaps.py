# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import json
import os
from unittest import TestCase

import matplotlib.colors
import numpy as np

from xcube.util.cmaps import Colormap, CUSTOM_CATEGORY
from xcube.util.cmaps import ColormapCategory
from xcube.util.cmaps import ColormapRegistry
from xcube.util.cmaps import create_colormap_from_config
from xcube.util.cmaps import DEFAULT_CMAP_NAME
from xcube.util.cmaps import parse_cm_name
from xcube.util.cmaps import parse_cm_code
from xcube.util.cmaps import load_snap_cpd_colormap

registry = ColormapRegistry()


class ColormapRegistryTest(TestCase):
    def setUp(self) -> None:
        self.registry = ColormapRegistry()

    def test_get_cmap(self):
        cmap, colormap = self.registry.get_cmap("bone")
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("bone", colormap.cm_name)
        self.assertEqual("Sequential (2)", colormap.cat_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_invalid(self):
        cmap, colormap = self.registry.get_cmap("BONE")
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual(DEFAULT_CMAP_NAME, colormap.cm_name)
        self.assertEqual("Perceptually Uniform Sequential", colormap.cat_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_alpha(self):
        cmap, colormap = self.registry.get_cmap("bone_alpha")
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("Sequential (2)", colormap.cat_name)
        self.assertEqual("bone", colormap.cm_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_reversed(self):
        cmap, colormap = self.registry.get_cmap("plasma_r")
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("plasma", colormap.cm_name)
        self.assertEqual("Perceptually Uniform Sequential", colormap.cat_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_reversed_alpha(self):
        cmap, colormap = self.registry.get_cmap("plasma_r_alpha")
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("plasma", colormap.cm_name)
        self.assertEqual("Perceptually Uniform Sequential", colormap.cat_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_from_code_type_node_norm(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783472",'
            ' "colors": ['
            '[0.0, "#00000000"], '
            '[0.6, "#ff0000aa"], '
            '[1.0, "#ffffffff"]'
            "]}"
        )
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("ucb783472", colormap.cm_name)
        self.assertEqual(CUSTOM_CATEGORY.name, colormap.cat_name)
        self.assertEqual((0.0, 0.6, 1.0), colormap.values)

    def test_get_cmap_from_code_type_node_not_norm(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783482",'
            ' "colors": ['
            '[-1.2, "#00000000"], '
            '[0.0, "#ff0000aa"], '
            '[1.2, "#ffffffff"]'
            "]}"
        )
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("ucb783482", colormap.cm_name)
        self.assertEqual(CUSTOM_CATEGORY.name, colormap.cat_name)
        self.assertEqual((-1.2, 0.0, 1.2), colormap.values)

    def test_get_cmap_from_code_type_key(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783473",'
            ' "colors": ['
            '[1, "#00000000"], '
            '[2, "#ff0000aa"], '
            '[5, "#ffffffff"]'
            '], "type": "categorical"}'
        )
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("ucb783473", colormap.cm_name)
        self.assertEqual("categorical", colormap.cm_type)
        self.assertEqual(CUSTOM_CATEGORY.name, colormap.cat_name)
        self.assertEqual([1, 2, 3, 5, 6], colormap.values)

    def test_get_cmap_from_code_type_bound(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783474",'
            ' "colors": ['
            '[0.0, "#00000000"], '
            '[0.6, "#ff0000aa"], '
            '[1.0, "#ffffffff"]'
            '], "type": "stepwise"}'
        )
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("ucb783474", colormap.cm_name)
        self.assertEqual("stepwise", colormap.cm_type)
        self.assertEqual(CUSTOM_CATEGORY.name, colormap.cat_name)
        self.assertEqual((0.0, 0.6, 1.0), colormap.values)

    def test_get_cmap_from_code_reversed_alpha(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783475_r_alpha",'
            ' "colors": ['
            '[0.0, "#00000000"], '
            '[0.5, "#ff0000aa"], '
            '[1.0, "#ffffffff"]'
            "]}"
        )
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(cmap(np.linspace(0, 1, 10)), np.ndarray)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("ucb783475", colormap.cm_name)
        self.assertEqual("continuous", colormap.cm_type)
        self.assertEqual(CUSTOM_CATEGORY.name, colormap.cat_name)
        self.assertEqual((0.0, 0.5, 1.0), colormap.values)

    def test_get_cmap_from_code_invalid(self):
        # User-defined color bars, #975
        cmap, colormap = self.registry.get_cmap(
            '{"name": "ucb783475_r_alpha", "colors": [[0.0, "#0000'
        )
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("Reds", colormap.cm_name)
        self.assertEqual("continuous", colormap.cm_type)
        self.assertEqual("Sequential", colormap.cat_name)
        self.assertIsNone(colormap.values)

    def test_get_cmap_num_colors(self):
        cmap, colormap = self.registry.get_cmap("plasma", num_colors=32)
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertEqual(32, cmap.N)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("plasma", colormap.cm_name)
        self.assertEqual("Perceptually Uniform Sequential", colormap.cat_name)
        self.assertEqual(256, colormap.cmap.N)

    def test_categories(self):
        categories = self.registry.categories
        self.assertIsInstance(categories, dict)
        self.assertGreaterEqual(len(categories), 8)
        self.assertIn("Perceptually Uniform Sequential", categories)
        self.assertIn("Sequential", categories)
        self.assertIn("Sequential (2)", categories)
        self.assertIn("Diverging", categories)
        self.assertIn("Qualitative", categories)
        self.assertIn("Cyclic", categories)
        self.assertIn("Ocean", categories)
        self.assertIn("Miscellaneous", categories)
        self.assertNotIn("Custom", categories)

    def test_category_descr(self):
        category = self.registry.categories.get("Perceptually Uniform Sequential")
        self.assertEqual(
            "For many applications, a perceptually uniform colormap"
            " is the best choice -"
            " one in which equal steps in data are perceived as equal"
            " steps in the color"
            " space",
            category.desc,
        )

    def test_colormaps(self):
        colormap = self.registry.colormaps.get("viridis")
        self.assertEqual("Perceptually Uniform Sequential", colormap.cat_name)
        self.assertEqual("viridis", colormap.cm_name)
        self.assertEqual(
            "iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAA"
            "xWXB3AAAAxElEQVR4nI2TQZLEIAwDW+Jr84T5/1"
            "PCHGzAkGRrDynZbclwIPro25FARg7FQhLYkBr9m"
            "o/M9Fb/9K6vH32ddwnMHz3g5GKx9A2+dPirl91j"
            "7pmpOvqqT36OM54z//XfGO/zVff3OXfv2w42Vnr"
            "O+V6rME3NGYPlU9v0qAFP3vOZ9MlcPJVZHXP0ya"
            "zrgQVvr31kGis76qYLc6guWuaWrh3hjf3Bw7d45"
            "mqdO+rexrhLz7NHDS1/mwY0KWthKRQnN0Y0mR+s"
            "DYkJKNzLygAAAABJRU5ErkJggg==",
            colormap.cmap_png_base64,
        )

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
        self.assertEqual(
            "iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAA"
            "xWXB3AAAAz0lEQVR4nHWSyZUEIQxDv0woHULnn1"
            "p5DiWDgekDz7IkxKrx+aYIUA0BgapvmhiT06zDW"
            "nED2SvrW2UQzRMMgh3HhQ+UTlfgXXhAiKsKCCUB"
            "SGktkUp/OR0eTY+r++6vbLV15LV/6rDnXf62Bn1"
            "fbd9n/jXvrOVrZ7xqOyfZcnvfc08+fWeVdXLPxX"
            "V+9no2Xdr1u09QongWVqJYmHjPRucjfcENz4rn4"
            "cs2Fnv9yb2PlPUwUxfZdELNs3CuRzP2vBDpj/sf"
            "nzNneXL2TFz8H4K3dTwq9eoAAAAAAElFTkSuQmC"
            "C",
            colormap.cmap_png_base64,
        )

    def test_load_snap_cpd_colormap(self):
        cmap_name = os.path.join(os.path.dirname(__file__), "chl_DeM2_200.cpd")
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

    def test_load_snap_cpd_colormap_with_alpha(self):
        cmap_name = os.path.join(os.path.dirname(__file__), "transparent_red.cpd")
        colormap = load_snap_cpd_colormap(cmap_name)
        self.assertIsInstance(colormap, Colormap)
        self.assertEqual("transparent_red", colormap.cm_name)
        self.assertIsInstance(
            colormap.cmap,
            matplotlib.colors.LinearSegmentedColormap,
        )
        self.assertIsInstance(
            colormap.norm,
            matplotlib.colors.Normalize,
        )

    def test_load_snap_cpd_colormap_invalid(self):
        cmap_name = os.path.join(
            os.path.dirname(__file__), "chl_DeM2_200_invalid_for_testing.cpd"
        )
        with self.assertRaises(ValueError):
            load_snap_cpd_colormap(cmap_name)

    def test_load_snap_cpd_colormap_missing(self):
        cmap_name = "test/webapi/im/chl_DeM2_200_not_existing.cpd"
        with self.assertRaises(FileNotFoundError):
            load_snap_cpd_colormap(cmap_name)


class ColormapTest(TestCase):
    def setUp(self) -> None:
        self.colormap = Colormap("coolwarm", cat_name="Diverging")

    def test_names(self):
        self.assertEqual("Diverging", self.colormap.cat_name)
        self.assertEqual("coolwarm", self.colormap.cm_name)
        self.assertEqual("continuous", self.colormap.cm_type)

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
            "iVBORw0KGgoAAAANSUhEUgAAAQAAAAABCAYAAAAxWXB3AAAA"
            "u0lEQVR4nJWMwQEDIAwCScbqDB26e7VCHxpN1D76gsARezxf"
            "MjOYO9y7xm3j9nxfWYMnb27Hxuc29/EfV255LN5s45F2GL9R"
            "d4bJHd5QuJqhZpuvmbZep0fOB490I35osdD1nrt8X7yHB88+"
            "cg0v/tDuEZkqAyUFS2YSkLSw4pFBhDHYlYHxZ/m9E8/s6Eho"
            "KKSeT6b2U6m5FysXe01OkPYsdX94tpr3e2RNYPCNvWuJawJJ"
            "6C3wo7u+hS8PX1PsPRnqDgAAAABJRU5ErkJggg==",
            base64,
        )


class ColormapParseTest(TestCase):
    def test_parse_cm_name(self):
        self.assertEqual(("viridis", False, False), parse_cm_name("viridis"))
        self.assertEqual(("viridis", True, False), parse_cm_name("viridis_r"))
        self.assertEqual(("viridis", False, True), parse_cm_name("viridis_alpha"))
        self.assertEqual(("viridis", True, True), parse_cm_name("viridis_r_alpha"))

    # User-defined color bars, #975
    def test_parse_cm_code(self):
        cm_name, cmap = parse_cm_code(
            '{"name": "ucb783473",'
            ' "colors": ['
            '[0.0, "#00000000"], '
            '[1.0, "#ffffffff"]'
            "]}"
        )
        self.assertEqual("ucb783473", cm_name)
        self.assertIsInstance(cmap, Colormap)
        self.assertEqual("ucb783473", cmap.cm_name)
        colors = cmap.cmap(np.array([0, 0.5, 1]))
        self.assertEqual([0.0, 0.0, 0.0, 0.0], list(colors[0]))
        self.assertEqual([1.0, 1.0, 1.0, 1.0], list(colors[2]))

        cm_name, cmap = parse_cm_code("{}")
        self.assertEqual("Reds", cm_name)
        self.assertEqual(None, cmap)


class ColormapConfigTest(TestCase):
    # User-defined color bars via xcube server config file, #1055
    def test_create_colormap_from_config_color_entry_object(self):
        cmap_config = dict(
            Identifier="my_cmap",
            Type="continuous",
            Colors=[
                dict(Value=0, Color="red", Label="low"),
                dict(Value=12, Color="#0000FF", Label="medium"),
                dict(Value=24, Color=[0, 1, 0, 0.3], Label="high"),
            ],
        )
        cmap, config_parse = create_colormap_from_config(cmap_config)
        self.assertIsInstance(cmap, Colormap)
        self.assertIsInstance(cmap.cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertEqual("continuous", cmap.cm_type)
        self.assertCountEqual([0, 12, 24], cmap.values)
        colors = cmap.cmap(np.array([0, 0.5, 1]))
        self.assertEqual([1.0, 0.0, 0.0, 1.0], list(colors[0]))
        self.assertEqual(
            {
                "name": "my_cmap",
                "type": "continuous",
                "colors": [
                    [0.0, "red", "low"],
                    [12.0, "#0000FF", "medium"],
                    [24.0, "#00ff004c", "high"],
                ],
            },
            config_parse,
        )

    def test_create_colormap_from_config_color_entry_tuple(self):
        cmap_config = dict(
            Identifier="my_cmap",
            Type="categorical",
            Colors=[
                [0, "red", "low"],
                [1, "#0000FF"],
                [2, [0, 1, 0], "high"],
            ],
        )
        cmap, config_parse = create_colormap_from_config(cmap_config)
        self.assertIsInstance(cmap, Colormap)
        self.assertIsInstance(cmap.cmap, matplotlib.colors.ListedColormap)
        self.assertEqual("categorical", cmap.cm_type)
        self.assertCountEqual([0, 1, 2, 3], cmap.values)
        colors = cmap.cmap(np.array([0, 1, 2]))
        self.assertEqual([1.0, 0.0, 0.0, 1.0], list(colors[0]))
        self.assertEqual([0.0, 0.0, 1.0, 1.0], list(colors[1]))
        self.assertEqual([0.0, 1.0, 0.0, 1.0], list(colors[2]))
        self.assertEqual(
            {
                "name": "my_cmap",
                "type": "categorical",
                "colors": [
                    [0.0, "red", "low"],
                    [1.0, "#0000FF"],
                    [2.0, "#00ff00ff", "high"],
                ],
            },
            config_parse,
        )
