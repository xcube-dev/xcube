# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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
import os
import re
from functools import cached_property
from typing import Dict, Tuple, List, Optional, Set

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
from PIL import Image

from xcube.constants import LOG

try:
    import cmocean.cm as ocm
except ImportError:
    ocm = None

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


# Have colormaps separated into categories:
# (taken from http://matplotlib.org/examples/color/colormaps_reference.html)
# colormaps for ocean:
# (taken from https://matplotlib.org/cmocean/)


class ColormapCategory:
    def __init__(self,
                 name: str,
                 desc: str,
                 cmap_names: Optional[Set[str]] = None):
        self.name = name
        self.desc = desc
        self.cmap_names = cmap_names


PERCEPTUALLY_UNIFORM_SEQUENTIAL = ColormapCategory(
    'Perceptually Uniform Sequential',
    'For many applications, a perceptually uniform colormap'
    ' is the best choice - one in which equal steps in data are'
    ' perceived as equal steps in the color space',
    {'viridis', 'inferno', 'plasma', 'magma'}
)

SEQUENTIAL_1 = ColormapCategory(
    'Sequential 1',
    'These colormaps are approximately monochromatic colormaps varying'
    ' smoothly between two color tones - usually from low saturation'
    ' (e.g. white) to high saturation (e.g. a bright blue).'
    ' Sequential colormaps are ideal for representing most scientific'
    ' data since they show a clear progression from'
    ' low-to-high values.',
    {'Blues', 'BuGn', 'BuPu',
     'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
     'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
     'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'}
)

SEQUENTIAL_2 = ColormapCategory(
    'Sequential 2',
    'Many of the values from the Sequential 2 plots are monotonically'
    ' increasing.',
    {'afmhot', 'autumn', 'bone', 'cool',
     'copper', 'gist_heat', 'hot',
     'pink', 'spring', 'summer', 'winter'}
)

DIVERGING = ColormapCategory(
    'Diverging',
    'These colormaps have a median value (usually light in color) and vary'
    ' smoothly to two different color tones at high and low values.'
    ' Diverging colormaps are ideal when your data has a median value'
    ' that is significant (e.g.  0, such that positive and negative'
    ' values are represented by different colors of the colormap).',
    {'BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
     'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
     'seismic'}
)
QUALITATIVE = ColormapCategory(
    'Qualitative',
    'These colormaps vary rapidly in color.'
    ' Qualitative colormaps are useful for choosing'
    ' a set of discrete colors.',
    {'Accent', 'Dark2', 'Paired', 'Pastel1',
     'Pastel2', 'Set1', 'Set2', 'Set3'}
)

UNCATEGORIZED = ColormapCategory(
    'Uncategorized',
    'Colormaps that don\'t fit into the categories above'
    ' or haven\'t been categorized yet.'
)

OCEAN = ColormapCategory(
    'Ocean',
    'Colormaps for commonly-used oceanographic variables'
    ' (from module cmocean.cm).'
)

CUSTOM = ColormapCategory(
    'Custom Colormaps',
    'Custom colormaps, e.g. loaded from SNAP *.cpd files.',
)

_CATEGORIES: Dict[str, ColormapCategory] = {
    c.name: c
    for c in (
        PERCEPTUALLY_UNIFORM_SEQUENTIAL,
        SEQUENTIAL_1,
        SEQUENTIAL_2,
        DIVERGING,
        QUALITATIVE,
        UNCATEGORIZED,
        OCEAN,
        CUSTOM,
    )
}

_CMAP_NAME_TO_CAT = {
    cm_name: cat
    for cat in _CATEGORIES.values()
    if cat.cmap_names
    for cm_name in cat.cmap_names
}


class Colormap:
    def __init__(self,
                 cat_name: str,
                 cmap_name: str,
                 cmap: matplotlib.colors.Colormap,
                 cmap_alpha: Optional[matplotlib.colors.Colormap] = None,
                 cmap_reverse: Optional[matplotlib.colors.Colormap] = None):
        self._cat_name = cat_name
        self._cmap_name = cmap_name
        self._cmap = cmap
        self._cmap_alpha = cmap_alpha
        self._cmap_reverse = cmap_reverse
        self._cmap_png_base64: Optional[str] = None

    @property
    def cat_name(self) -> str:
        return self._cat_name

    @property
    def cmap_name(self) -> str:
        return self._cmap_name    \

    @property
    def cmap(self) -> matplotlib.colors.Colormap:
        return self._cmap

    @cached_property
    def cmap_alpha(self) -> matplotlib.colors.Colormap:
        if self._cmap_alpha is None:
            self._cmap_alpha = _get_alpha_cmap(self.cmap)
        return self._cmap_alpha

    @cached_property
    def cmap_reverse(self) -> matplotlib.colors.Colormap:
        if self._cmap_reverse is None:
            self._cmap_reverse = self.cmap.reversed(
                name=self._cmap_name + "_r"
            )
        return self._cmap_reverse

    @cached_property
    def cmap_png_base64(self) -> str:
        if self._cmap_png_base64 is None:
            self._cmap_png_base64 = _get_cmap_png_base64(self.cmap)
        return self._cmap_png_base64


# def get_cmap(cmap_name: str,
#              default_cmap_name: str = 'viridis',
#              num_colors: Optional[int] = None) -> CategorizedColormaps:

class CategorizedColormaps:
    def __init__(self):
        pass

    def _load(self):
        categories: Dict[str, ColormapCategory] = {}
        cmaps: Dict[str, Colormap] = {}

        # Add standard Matplotlib colormaps
        for cm_name, cmap in matplotlib.colormaps.items():
            template_category = _CMAP_NAME_TO_CAT.get(cm_name)
            if template_category is None:
                template_category = UNCATEGORIZED

            category = categories.get(template_category.name)
            if category is None:
                category = ColormapCategory(template_category.name,
                                            template_category.desc,
                                            set())

                categories[template_category.name] = category

            category.cmap_names.add(cm_name)
            cmaps[cm_name] = Colormap(category.name, cm_name, cmap)

        # Add Ocean colormaps, if any
        if ocm is not None:
            ocean_cat_template = OCEAN
            ocean_cat_desc = _CAT_NAME_TO_DESC[ocean_cat_name]
            ocean_cm_names = [k for k in ocm.__dict__.keys()
                              if isinstance(getattr(ocm, k),
                                            matplotlib.colors.Colormap)
                              and not k.endswith("_r")]
            categories[ocean_cat_name] = ocean_cat_desc, ocean_cm_names

        # Add custom colormaps, if any
        if custom_cmaps:
            custom_cat_name = CUSTOM[0]
            custom_cat_desc = _CAT_NAME_TO_DESC[ocean_cat_name]
            custom_cmap_names = []
            for custom_cmap in custom_cmaps:
                if not os.path.isfile(custom_cmap):
                    LOG.error(f"Missing custom colormap file:"
                              f" {custom_cmaps}")
                    continue
                try:
                    cmap = _load_snap_colormap(custom_cmap)
                except ValueError as e:
                    LOG.warning(f'Detected invalid custom colormap'
                                f' {custom_cmap!r}: {e}',
                                exc_info=True)
                custom_cmap_names.append(cmap)
                cbar_list.append((cmap_name, _get_cmap_png_base64(cmap)))

                cbar_list.append((cmap_name, _get_cmap_png_base64(cmap)))
                cbar_list.append((new_name, _get_cmap_png_base64(new_cmap)))

            new_cmaps.append(
                (cmap_category, cmap_description, tuple(cbar_list)))

        return new_cmaps


def _get_alpha_cmap(cmap: matplotlib.colors.Colormap) \
        -> Optional[matplotlib.colors.Colormap]:
    """Add extra colormaps with alpha gradient.
    See http://matplotlib.org/api/colors_api.html
    """
    cmap_name = cmap.name + '_alpha' \
        if hasattr(cmap, 'name') else 'unknown'

    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        new_segmentdata = dict(cmap._segmentdata)
        # let alpha increase from 0.0 to 0.5
        new_segmentdata['alpha'] = ((0.0, 0.0, 0.0),
                                    (0.5, 1.0, 1.0),
                                    (1.0, 1.0, 1.0))
        return matplotlib.colors.LinearSegmentedColormap(
            cmap_name, new_segmentdata
        )

    if isinstance(cmap, matplotlib.colors.ListedColormap):
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
        return matplotlib.colors.ListedColormap(
            new_colors, name=cmap_name
        )

    LOG.warning(
        f'Could not create alpha colormap for cmap of type {type(cmap)}'
    )

    return None


def _get_cmap_png_base64(cmap: matplotlib.colors.Colormap) -> str:
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

    return cbar_png_bytes


def _load_snap_colormap(cpd_file_path: str):
    points, log_scaled = _parse_snap_cpd_file(cpd_file_path)
    samples = [sample for sample, _ in points]
    # Uncomment, as soon as we know how to deal with this:
    # if log_scaled:
    #     norm = matplotlib.colors.LogNorm(min(samples), max(samples))
    # else:
    #     norm = matplotlib.colors.Normalize(min(samples), max(samples))
    norm = matplotlib.colors.Normalize(min(samples), max(samples))
    colors = list(zip(map(norm, samples),
                      ['#%02x%02x%02x' % color for _, color in points]))
    cpd_name, _ = os.path.splitext(os.path.basename(cpd_file_path))
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        cpd_name, colors
    )


Sample = float
Color = Tuple[int, int, int]
Palette = List[Tuple[Sample, Color]]
LogScaled = bool


def _parse_snap_cpd_file(cpd_file_path: str) \
        -> Tuple[Palette, LogScaled]:
    with open(cpd_file_path, "r") as f:

        illegal_format_msg = f"Illegal SNAP *.cpd format: {cpd_file_path}"

        entries: Dict[str, str] = dict()
        for line in f.readlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    k, v = line.split("=", maxsplit=1)
                except ValueError:
                    raise ValueError(illegal_format_msg)
                entries[k.strip()] = v.strip()

        for keyword in ("autoDistribute", "isLogScaled"):
            if keyword in entries:
                LOG.warning(f"Unrecognized keyword {keyword!r}"
                            f" in SNAP *.cpd file {cpd_file_path}")

        # Uncomment, as soon as we know how to deal with this:
        # log_scaled = entries.get("isLogScaled") == "true"
        log_scaled = False

        try:
            num_points = int(entries.get("numPoints"))
        except ValueError:
            raise ValueError(illegal_format_msg)

        points = []
        for i in range(num_points):
            try:
                r, g, b = map(int, entries.get(f"color{i}", "").split(","))
                sample = float(entries.get(f"sample{i}"))
            except ValueError:
                raise ValueError(illegal_format_msg)
            points.append((sample, (r, g, b)))

        return points, log_scaled


def _get_color(colortext):
    f = open(colortext, "r")
    lines = f.readlines()
    c = []
    if any('color' in line for line in lines):
        for line in lines:
            if "color" in line:
                r, g, b = (
                    ((re.split(r'\W+', line, 1)[1:])[0].strip()).split(','))
                hex_col = ('#%02x%02x%02x' % (int(r), int(g), int(b)))
                c.append(hex_col)
    else:
        LOG.warning('Keyword "color" not found. SNAP .cpd file invalid.')
        return
    f.close()
    return c


def get_tick_val_col(colortext):
    f = open(colortext, "r")
    lines = f.readlines()
    values = []
    if any('sample' in line for line in lines):
        for line in lines:
            if "sample" in line:
                value = ((re.split(r'\W+', line, 1)[1:])[0].strip())
                values.append(float(value))
    else:
        LOG.warning('Keyword "sample" not found. SNAP .cpd file invalid.')
        return
    f.close()
    return values


def get_norm(colortext):
    values = get_tick_val_col(colortext)
    norm = matplotlib.colors.LogNorm(min(values), max(values))
    return norm, values


def _check_if_exists(SNAP_CPD_LIST):
    valid_path = []
    for item in SNAP_CPD_LIST:
        if os.path.isfile(item):
            valid_path.append(item)
    return tuple(valid_path)
