# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import base64
import io
import logging
from threading import Lock

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
from PIL import Image

try:
    import cmocean.cm as ocm
except ImportError:
    ocm = None

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

_LOG = logging.getLogger('xcube')

# Have colormaps separated into categories:
# (taken from http://matplotlib.org/examples/color/colormaps_reference.html)
# colormaps for ocean:
# (taken from https://matplotlib.org/cmocean/)

_CMAPS = (('Perceptually Uniform Sequential',
           'For many applications, a perceptually uniform colormap is the best choice - '
           'one in which equal steps in data are perceived as equal steps in the color space',
           ('viridis', 'inferno', 'plasma', 'magma')),
          ('Sequential 1',
           'These colormaps are approximately monochromatic colormaps varying smoothly '
           'between two color tones - usually from low saturation (e.g. white) to high '
           'saturation (e.g. a bright blue). Sequential colormaps are ideal for '
           'representing most scientific data since they show a clear progression from '
           'low-to-high values.',
           ('Blues', 'BuGn', 'BuPu',
            'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
            'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
            'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd')),
          ('Sequential 2',
           'Many of the values from the Sequential 2 plots are monotonically increasing.',
           ('afmhot', 'autumn', 'bone', 'cool',
            'copper', 'gist_heat', 'gray', 'hot',
            'pink', 'spring', 'summer', 'winter')),
          ('Diverging',
           'These colormaps have a median value (usually light in color) and vary '
           'smoothly to two different color tones at high and low values. Diverging '
           'colormaps are ideal when your data has a median value that is significant '
           '(e.g.  0, such that positive and negative values are represented by '
           'different colors of the colormap).',
           ('BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
            'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
            'seismic')),
          ('Qualitative',
           'These colormaps vary rapidly in color. Qualitative colormaps are useful for '
           'choosing a set of discrete colors.',
           ('Accent', 'Dark2', 'Paired', 'Pastel1',
            'Pastel2', 'Set1', 'Set2', 'Set3')),
          ('Ocean',
           'Colormaps for commonly-used oceanographic variables. ',
           ('thermal', 'haline', 'solar', 'ice', 'gray',
            'oxy', 'deep', 'dense', 'algae',
            'matter', 'turbid', 'speed', 'amp', 'tempo',
            'phase', 'balance', 'delta', 'curl')),
          ('Miscellaneous',
           'Colormaps that don\'t fit into the categories above.',
           ('gist_earth', 'terrain', 'ocean', 'gist_stern',
            'brg', 'CMRmap', 'cubehelix',
            'gnuplot', 'gnuplot2', 'gist_ncar',
            'nipy_spectral', 'jet', 'rainbow',
            'gist_rainbow', 'hsv', 'flag', 'prism')))

_CBARS_LOADED = False
_LOCK = Lock()


def get_cmaps():
    """
    Return a JSON-serializable tuple containing records of the form:
     (<cmap-category>, <cmap-category-description>, <cmap-tuples>),
    where <cmap-tuples> is a tuple containing records of the form (<cmap-name>, <cbar-png-bytes>), and where
    <cbar-png-bytes> are encoded PNG images of size 256 x 2 pixels,
    :return: all known matplotlib color maps
    """

    global _CMAPS
    ensure_cmaps_loaded()
    return _CMAPS


def ensure_cmaps_loaded():
    """
    Loads all color maps from matplotlib and registers additional ones, if not done before.
    """
    global _CBARS_LOADED, _CMAPS
    if not _CBARS_LOADED:
        _LOCK.acquire()
        if not _CBARS_LOADED:
            new_cmaps = []
            for cmap_category, cmap_description, cmap_names in _CMAPS:
                if cmap_category == 'Ocean' and ocm is None:
                    continue
                cbar_list = []
                for cmap_name in cmap_names:
                    try:
                        if cmap_category == 'Ocean':
                            cmap = getattr(ocm, cmap_name)
                        else:
                            cmap = cm.get_cmap(cmap_name)
                    except ValueError:
                        _LOG.warning('detected invalid colormap "%s"' % cmap_name)
                        continue
                    # Add extra colormaps with alpha gradient
                    # see http://matplotlib.org/api/colors_api.html
                    if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
                        new_name = cmap.name + '_alpha'
                        new_segmentdata = dict(cmap._segmentdata)
                        # let alpha increase from 0.0 to 0.5
                        new_segmentdata['alpha'] = ((0.0, 0.0, 0.0),
                                                    (0.5, 1.0, 1.0),
                                                    (1.0, 1.0, 1.0))
                        new_cmap = matplotlib.colors.LinearSegmentedColormap(new_name, new_segmentdata)
                        cm.register_cmap(cmap=new_cmap)
                    elif type(cmap) == matplotlib.colors.ListedColormap:
                        new_name = cmap.name + '_alpha'
                        new_colors = list(cmap.colors)
                        a_slope = 2.0 / cmap.N
                        a = 0
                        for i in range(len(new_colors)):
                            new_color = new_colors[i]
                            if not isinstance(new_color, str):
                                if len(new_color) == 3:
                                    r, g, b = new_color
                                    new_colors[i] = r, g, b, a
                                elif len(new_color) == 4:
                                    r, g, b, a_old = new_color
                                    new_colors[i] = r, g, b, min(a, a_old)
                            a += a_slope
                            if a > 1.0:
                                a = 1.0
                        new_cmap = matplotlib.colors.ListedColormap(new_colors, name=new_name)
                        cm.register_cmap(cmap=new_cmap)
                    else:
                        new_name = cmap.name + '_alpha' if hasattr(cmap, 'name') else 'unknown'
                        _LOG.warning('could not create colormap "{}" because "{}" is of unknown type {}'
                                     .format(new_name, cmap.name, type(cmap)))

                    gradient = np.linspace(0, 1, 256)
                    gradient = np.vstack((gradient, gradient))
                    image_data = cmap(gradient, bytes=True)
                    image = Image.fromarray(image_data, 'RGBA')

                    # ostream = io.FileIO('../cmaps/' + cmap_name + '.png', 'wb')
                    # image.save(ostream, format='PNG')
                    # ostream.close()

                    ostream = io.BytesIO()
                    image.save(ostream, format='PNG')
                    cbar_png_bytes = ostream.getvalue()
                    ostream.close()

                    cbar_png_data = base64.b64encode(cbar_png_bytes)
                    cbar_png_bytes = cbar_png_data.decode('unicode_escape')

                    cbar_list.append((cmap_name, cbar_png_bytes))
                new_cmaps.append((cmap_category, cmap_description, tuple(cbar_list)))
            _CMAPS = tuple(new_cmaps)
            _CBARS_LOADED = True
            # import pprint
            # pprint.pprint(_CMAPS)
        _LOCK.release()
