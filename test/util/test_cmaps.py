import os
from unittest import TestCase

import matplotlib.cm as cm
from matplotlib.colors import Colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

from xcube.util.cmaps import _load_snap_colormap
from xcube.util.cmaps import ensure_cmaps_loaded
from xcube.util.cmaps import get_cmap
from xcube.util.cmaps import get_cmaps


class CmapsTest(TestCase):

    def test_get_cmap(self):

        cmap_name, cmap = get_cmap('plasma')
        self.assertEqual('plasma', cmap_name)
        self.assertIsInstance(cmap, Colormap)

        cmap_name, cmap = get_cmap('PLASMA')
        self.assertEqual('viridis', cmap_name)
        self.assertIsInstance(cmap, Colormap)

        cmap_name, cmap = get_cmap('PLASMA', default_cmap_name='magma')
        self.assertEqual('magma', cmap_name)
        self.assertIsInstance(cmap, Colormap)

        with self.assertRaises(ValueError):
            get_cmap('PLASMA', default_cmap_name='MAGMA')

    def test_get_cmaps_returns_singleton(self):
        cmaps = get_cmaps()
        self.assertIs(cmaps, get_cmaps())
        self.assertIs(cmaps, get_cmaps())

    def test_get_cmaps_registers_ocean_colour(self):
        cmap = cm.get_cmap('deep', 256)
        self.assertTrue((type(cmap) is LinearSegmentedColormap) or (
                type(cmap) is ListedColormap))

    def test_get_cmaps_retruns_equal_size_recs(self):
        cmaps = get_cmaps()
        rec_len = len(cmaps[0])
        self.assertEqual(rec_len, 3)
        for cmap in cmaps:
            self.assertEqual(len(cmap), rec_len)

    def test_get_cmaps_categories(self):
        cmaps = get_cmaps()
        self.assertGreaterEqual(len(cmaps), 8)
        self.assertEqual(cmaps[0][0], 'Perceptually Uniform Sequential')
        self.assertEqual(cmaps[1][0], 'Sequential 1')
        self.assertEqual(cmaps[2][0], 'Sequential 2')
        self.assertEqual(cmaps[3][0], 'Diverging')
        self.assertEqual(cmaps[4][0], 'Qualitative')
        self.assertEqual(cmaps[5][0], 'Ocean')
        self.assertEqual(cmaps[6][0], 'Miscellaneous')
        self.assertEqual(cmaps[7][0], 'Custom Colormaps')

    def test_get_cmaps_category_descr(self):
        cmaps = get_cmaps()
        self.assertEqual(
            cmaps[0][1],
            'For many applications, a perceptually uniform colormap'
            ' is the best choice -'
            ' one in which equal steps in data are perceived as equal'
            ' steps in the color'
            ' space'
        )

    def test_get_cmaps_category_tuples(self):
        cmaps = get_cmaps()
        category_tuple = cmaps[0][2]
        self.assertEqual(len(category_tuple), 8)
        self.assertEqual(category_tuple[0][0], 'viridis')
        self.assertEqual(category_tuple[0][1],
                         'iVBORw0KGgoAAAANSUhEUgAAAQAAAAACCAYAAAC3zQLZ'
                         'AAAAzklEQVR4nO2TQZLFIAhEX7dXmyPM/Y8SZwEqMcnU'
                         '3/9QZTU8GszC6Ee/HQlk5FAsJIENqVGv/piZ3uqf3nX6'
                         'Vtd+l8D8UwNOLhZL3+BLh796OXvMdWaqtrrqnZ/tjvuZ'
                         'T/0XxnN/5f25z9X7tIMTKzV7/5yrME3NHoPlUzvplgOe'
                         'vOcz6ZO5eCqzOmark1nHDQveHuuYaazZkTcdmE110HJu'
                         '6doR3tgfPHyL51zNc0fd2xjf0vPukUPL36YBTcpcWArF'
                         'yY0RTca88cYbXxt/gUOJC8yRF1kAAAAASUVORK5CYII=')

        self.assertEqual(category_tuple[1][0], 'viridis_alpha')
        self.assertEqual(category_tuple[2][0], 'inferno')
        self.assertEqual(category_tuple[3][0], 'inferno_alpha')

    def test_cmocean_category(self):
        cmaps = get_cmaps()
        category_tuple = cmaps[5][2]
        self.assertTrue(len(category_tuple) >= 34)
        self.assertEqual(category_tuple[0][0], 'thermal')
        self.assertEqual(category_tuple[0][1],
                         'iVBORw0KGgoAAAANSUhEUgAAAQAAAAACCAYAAAC3zQLZ'
                         'AAAA2klEQVR4nO2S6xHDMAiDP+FROkL3Xy30RwBju52g'
                         '8V0OIcnyKxqvtwsD5SfAUPZNE6M4VR2hJTdQeBX6UhlY'
                         '8xgDY8V24A15pMuIXcQHJo4qwOQYIHlojpT6zWnzqDxR'
                         'o+/+zFZbR7H2Tx3WvMPf1qDvq+17zz/m7TV97YxHbefE'
                         'W27ve+7Oe9xZZu3cdXCdr17XokurfvYOcmTXxHJkE2P3'
                         '2ei8eVxww1WJecRlBxZr/cndj+T5MKULbzqm5pnY56MF'
                         'jnkmPH7cb7xXzvR49RRO3njGM57xt+MDC391Pt11tkYA'
                         'AAAASUVORK5CYII=')

        self.assertEqual(category_tuple[1][0], 'thermal_alpha')
        self.assertEqual(category_tuple[2][0], 'haline')
        self.assertEqual(category_tuple[3][0], 'haline_alpha')

    def test_get_cmaps_registers_snap_cpd_file(self):
        cmap_name = os.path.join(os.path.dirname(__file__), 'chl_DeM2_200.cpd')
        cmap = _load_snap_colormap(cmap_name)
        self.assertIsInstance(cmap, (LinearSegmentedColormap, ListedColormap))

    def test_get_cmaps_registers_invalid_snap_cpd_file(self):
        cmap_name = os.path.join(os.path.dirname(__file__),
                                 'chl_DeM2_200_invalid_for_testing.cpd')
        with self.assertRaises(ValueError):
            _load_snap_colormap(cmap_name)

    def test_get_cmaps_registers_non_existing_snap_cpd_file(self):
        cmap_name = 'test/webapi/im/chl_DeM2_200_not_existing.cpd'
        with self.assertRaises(FileNotFoundError):
            _load_snap_colormap(cmap_name)
