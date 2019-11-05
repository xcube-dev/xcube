import copy
from unittest import TestCase

import numpy as np
from PIL import Image
from matplotlib import cm


# * http://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
# * http://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap
# * http://stackoverflow.com/questions/10443295/combine-3-separate-numpy-arrays-to-an-rgb-image-in-python
# * http://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow-in-matplotlib
#
# * http://matplotlib.org/api/cm_api.html#module-matplotlib.cm
# * http://matplotlib.org/users/colormaps.html
# * http://matplotlib.org/api/colors_api.html#
# * https://bids.github.io/colormap/

class PILImageTest(TestCase):
    def test_array_to_rgba(self):
        nan = np.nan
        array = np.array([
            [0.0, 0.1, 0.2, nan],
            [0.2, 0.3, nan, 0.5],
            [0.4, nan, 0.6, 0.7],
            [nan, 0.7, 0.8, 0.9],
        ])
        # print(array, array.dtype)

        min = 0.1
        max = 0.8

        array.clip(min, max, out=array)
        # print(array, array.dtype)

        array -= min
        array *= 1.0 / (max - min)
        # print(array, array.dtype)

        array = np.ma.masked_invalid(array, copy=False)
        # print(array, array.dtype)

        # import pprint
        # pprint.pprint(dir(cm.gist_earth))

        # We want to have transparency where the array is masked out.
        # This is done via colormap.set_bad(color, alpha).
        # However, in doing we would modify the global cm.gist_earth singleton so we create a copy here:
        #
        cmap_copy = copy.copy(cm.gist_earth)
        cmap_copy.set_bad('k', 0)
        array = cmap_copy(array, bytes=True)
        # print(array, array.dtype)
        self.assertEqual([[[0, 0, 0, 255],
                           [0, 0, 0, 255],
                           [23, 64, 121, 255],
                           [0, 0, 0, 0],
                           ],

                          [[23, 64, 121, 255],
                           [48, 128, 126, 255],
                           [0, 0, 0, 0],
                           [130, 168, 83, 255],
                           ],

                          [[65, 149, 82, 255],
                           [0, 0, 0, 0],
                           [184, 179, 95, 255],
                           [212, 176, 147, 255],
                           ],

                          [[0, 0, 0, 0],
                           [212, 176, 147, 255],
                           [253, 250, 250, 255],
                           [253, 250, 250, 255],
                           ],
                          ], array.tolist())

        im = Image.fromarray(array, mode='RGBA')

        # im.show('Raw Image')

        self.assertEqual((0, 0, 0, 255), im.getpixel((0, 0)))
        self.assertEqual((23, 64, 121, 255), im.getpixel((2, 0)))
        self.assertEqual((0, 0, 0, 0), im.getpixel((1, 2)))
        self.assertEqual((212, 176, 147, 255), im.getpixel((1, 3)))
