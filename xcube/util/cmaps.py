# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import base64
import io
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Optional, Any, Union

import fsspec
import matplotlib
import matplotlib.colors
import numpy as np
from PIL import Image

from xcube.constants import LOG
from xcube.util.assertions import assert_instance, assert_given

try:
    # noinspection PyPackageRequirements
    import cmocean.cm as cmo
except ImportError:
    cmo = None

OCEAN_PREFIX = "cmo"
ALPHA_SUFFIX = "_alpha"
REVERSED_SUFFIX = "_r"  # Used by ocm / Ocean

DEFAULT_CMAP_NAME = "viridis"


# Have colormaps separated into categories taken from
# https://matplotlib.org/stable/gallery/color/colormap_reference.html
# colormaps for ocean taken from
# https://matplotlib.org/cmocean/


class ColormapCategory:
    def __init__(self, name: str, desc: str, cm_names: Optional[list[str]] = None):
        self.name = name
        self.desc = desc
        self.cm_names = cm_names


PERCEPTUALLY_UNIFORM_SEQUENTIAL_CATEGORY = ColormapCategory(
    "Perceptually Uniform Sequential",
    "For many applications, a perceptually uniform colormap"
    " is the best choice - one in which equal steps in data are"
    " perceived as equal steps in the color space",
    ["viridis", "plasma", "inferno", "magma", "cividis"],
)

SEQUENTIAL_CATEGORY = ColormapCategory(
    "Sequential",
    "These colormaps are approximately monochromatic colormaps varying"
    " smoothly between two color tones - usually from low saturation"
    " (e.g. white) to high saturation (e.g. a bright blue)."
    " Sequential colormaps are ideal for representing most scientific"
    " data since they show a clear progression from"
    " low-to-high values.",
    [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ],
)

SEQUENTIAL_2_CATEGORY = ColormapCategory(
    "Sequential (2)",
    "Many of the values from the Sequential 2 plots are monotonically" " increasing.",
    [
        "binary",
        "gist_yarg",
        "gist_gray",
        "gray",
        "bone",
        "pink",
        "spring",
        "summer",
        "autumn",
        "winter",
        "cool",
        "Wistia",
        "hot",
        "afmhot",
        "gist_heat",
        "copper",
    ],
)

DIVERGING_CATEGORY = ColormapCategory(
    "Diverging",
    "These colormaps have a median value (usually light in color) and vary"
    " smoothly to two different color tones at high and low values."
    " Diverging colormaps are ideal when your data has a median value"
    " that is significant (e.g.  0, such that positive and negative"
    " values are represented by different colors of the colormap).",
    [
        "PiYG",
        "PRGn",
        "BrBG",
        "PuOr",
        "RdGy",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "coolwarm",
        "bwr",
        "seismic",
    ],
)

QUALITATIVE_CATEGORY = ColormapCategory(
    "Qualitative",
    "These colormaps vary rapidly in color."
    " Qualitative colormaps are useful for choosing"
    " a set of discrete colors.",
    [
        "Pastel1",
        "Pastel2",
        "Paired",
        "Accent",
        "Dark2",
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "tab20",
        "tab20b",
        "tab20c",
    ],
)

CYCLIC_CATEGORY = ColormapCategory(
    "Cyclic", "These colormaps are cyclic.", ["twilight", "twilight_shifted", "hsv"]
)

MISCELLANEOUS_CATEGORY = ColormapCategory(
    "Miscellaneous",
    "Colormaps that don't fit into the categories above"
    " or haven't been categorized yet.",
    [
        "flag",
        "prism",
        "ocean",
        "gist_earth",
        "terrain",
        "gist_stern",
        "gnuplot",
        "gnuplot2",
        "CMRmap",
        "cubehelix",
        "brg",
        "gist_rainbow",
        "rainbow",
        "jet",
        "turbo",
        "nipy_spectral",
        "gist_ncar",
    ],
)

OCEAN_CATEGORY = ColormapCategory(
    "Ocean",
    "Colormaps for commonly-used oceanographic variables" " (from module cmocean.cm).",
)

CUSTOM_CATEGORY = ColormapCategory(
    "Custom",
    "Custom colormaps, e.g. loaded from SNAP *.cpd files.",
)

TEMPLATE_CATEGORIES: dict[str, ColormapCategory] = {
    c.name: c
    for c in (
        PERCEPTUALLY_UNIFORM_SEQUENTIAL_CATEGORY,
        SEQUENTIAL_CATEGORY,
        SEQUENTIAL_2_CATEGORY,
        DIVERGING_CATEGORY,
        QUALITATIVE_CATEGORY,
        CYCLIC_CATEGORY,
        OCEAN_CATEGORY,
        CUSTOM_CATEGORY,
        MISCELLANEOUS_CATEGORY,
    )
}

_CM_NAME_TO_TEMPLATE_CATEGORY = {
    cm_name: template_cat
    for template_cat in TEMPLATE_CATEGORIES.values()
    if template_cat.cm_names
    for cm_name in template_cat.cm_names
}


class Colormap:
    """Internal class (non-API) that represents a color mapping."""

    def __init__(
        self,
        cm_name: str,
        *,
        cm_type: Optional[str] = None,
        cat_name: Optional[str] = None,
        cmap: Optional[matplotlib.colors.Colormap] = None,
        cmap_reversed: Optional[matplotlib.colors.Colormap] = None,
        cmap_alpha: Optional[matplotlib.colors.Colormap] = None,
        norm: Optional[matplotlib.colors.Normalize] = None,
        values: Optional[Sequence[Union[int, float]]] = None,
    ):
        self._cm_name = cm_name
        self._cm_type = cm_type or "node"
        self._cat_name = cat_name
        self._cmap = cmap
        self._cmap_reversed = cmap_reversed
        self._cmap_alpha = cmap_alpha
        self._cmap_reversed_alpha = None
        self._cmap_png_base64: Optional[str] = None
        self._norm = norm
        self._colors = values

    @property
    def cm_name(self) -> str:
        """The colormap's name."""
        return self._cm_name

    @property
    def cm_type(self) -> str:
        """The colormap type, always one of "node", "bound", "key"."""
        return self._cm_type

    @property
    def cat_name(self) -> Optional[str]:
        """The colormap's category name."""
        return self._cat_name

    @cached_property
    def cmap(self) -> matplotlib.colors.Colormap:
        """The matplotlib colormap."""
        if self._cmap is None:
            self._cmap = matplotlib.colormaps[self.cm_name]
        return self._cmap

    @cached_property
    def cmap_alpha(self) -> matplotlib.colors.Colormap:
        """The matplotlib colormap with alpha-blending of smaller values."""
        if self._cmap_alpha is None:
            _, self._cmap_alpha = get_alpha_cmap(self.cm_name, self.cmap)
        return self._cmap_alpha

    @cached_property
    def cmap_reversed(self) -> matplotlib.colors.Colormap:
        """The reversed matplotlib colormap."""
        """Whether to reverse the color map."""
        if self._cmap_reversed is None:
            _, self._cmap_reversed = get_reverse_cmap(self.cm_name, self.cmap)
        return self._cmap_reversed

    @cached_property
    def cmap_reversed_alpha(self) -> matplotlib.colors.Colormap:
        """The reversed matplotlib colormap with alpha-blending of smaller values."""
        if self._cmap_reversed_alpha is None:
            cm_name, cmap_reversed = get_reverse_cmap(self.cm_name, self.cmap)
            _, self._cmap_reversed_alpha = get_alpha_cmap(cm_name, cmap_reversed)
        return self._cmap_reversed_alpha

    @cached_property
    def cmap_png_base64(self) -> str:
        """The base64 encoding of the colormap's PNG image."""
        if self._cmap_png_base64 is None:
            self._cmap_png_base64 = get_cmap_png_base64(self.cmap)
        return self._cmap_png_base64

    @property
    def norm(self) -> Optional[matplotlib.colors.Normalize]:
        """The matplotlib norm, if any."""
        return self._norm

    @property
    def values(self) -> Optional[Sequence[Union[int, float]]]:
        """The list of values that are mapped to colors."""
        return self._colors


class ColormapProvider(ABC):
    @abstractmethod
    def get_cmap(
        self, cm_name: str, num_colors: Optional[int] = None
    ) -> tuple[matplotlib.colors.Colormap, Colormap]:
        """Get a colormap for the given *cm_name*.

        If *cm_name* is not available, the method may choose another
        colormap and return its name.

        A colormap provider may support colormaps that are

        * reversed (*cm_name* with suffix `"_r"`),
        * have alpha blending (*cm_name* with suffix `"_alpha"`),
        * both (*cm_name* with suffix `"_r_alpha"`).

        Args:
            cm_name: Colormap name.
            num_colors: Optional number of colors, that is, the
                resolution of the colormap gradient.

        Returns:
            A tuple (cmap, colormap) comprising
            the colormap as an instance of
            ``matplotlib.colors.Colormap`` styled according to
            the given *cm_name* suffix(es), and the base colormap object.
        """


class ColormapRegistry(ColormapProvider):
    def __init__(self, *colormaps: Colormap):
        self._categories: dict[str, ColormapCategory] = {}
        self._colormaps: dict[str, Colormap] = {}
        # Add standard Matplotlib colormaps
        self._register_mpl_cmaps()
        # Add Ocean colormaps, if any
        self._register_ocm_cmaps()
        # Add custom colormaps, if any
        for colormap in colormaps:
            self.register_colormap(colormap)

    @property
    def categories(self) -> dict[str, ColormapCategory]:
        return self._categories

    @property
    def colormaps(self) -> dict[str, Colormap]:
        return self._colormaps

    def register_colormap(self, colormap: Colormap):
        cat_name = colormap.cat_name or MISCELLANEOUS_CATEGORY.name
        template_cat = TEMPLATE_CATEGORIES.get(cat_name)
        if template_cat is None:
            raise ValueError(f"Unknown colormap category {cat_name!r}")
        category = self._get_category(template_cat)
        category.cm_names.append(colormap.cm_name)
        self._colormaps[colormap.cm_name] = colormap

    def find_colormap(self, cm_name: str) -> Optional[Colormap]:
        assert_instance(cm_name, str, name="cm_name")
        # Strips suffixes from cm_name
        cm_name, _, _ = parse_cm_name(cm_name)
        return self._colormaps.get(cm_name)

    def get_cmap(
        self, cm_name: str, num_colors: Optional[int] = None
    ) -> tuple[matplotlib.colors.Colormap, Colormap]:
        assert_instance(cm_name, str, name="cm_name")
        if num_colors is not None:
            assert_instance(num_colors, int, name="num_colors")

        colormap: Optional[Colormap] = None
        if cm_name.startswith("{"):
            cm_name, colormap = parse_cm_code(cm_name)

        cm_name, reverse, alpha = parse_cm_name(cm_name)
        if colormap is None:
            colormap = self._colormaps.get(cm_name)

        if colormap is None:
            cm_name = DEFAULT_CMAP_NAME
            colormap = self._colormaps[cm_name]

        if reverse and alpha:
            cmap: matplotlib.colors.Colormap = colormap.cmap_reversed_alpha
        elif reverse:
            cmap: matplotlib.colors.Colormap = colormap.cmap_reversed
        elif alpha:
            cmap: matplotlib.colors.Colormap = colormap.cmap_alpha
        else:
            cmap: matplotlib.colors.Colormap = colormap.cmap
        if num_colors is not None:
            cmap = cmap.resampled(num_colors)
        return cmap, colormap

    def to_json(self) -> list:
        result = []
        # Loop through TEMPLATE_CATEGORIES to preserve category order
        for cat_name in TEMPLATE_CATEGORIES.keys():
            category = self._categories.get(cat_name)
            if category is not None:
                result.append(
                    [
                        category.name,
                        category.desc,
                        [
                            [cm_name, self._colormaps[cm_name].cmap_png_base64]
                            for cm_name in category.cm_names
                        ],
                    ]
                )
        return result

    def _get_category(self, template_cat: ColormapCategory):
        category = self._categories.get(template_cat.name)
        if category is None:
            category = ColormapCategory(
                template_cat.name, template_cat.desc, cm_names=[]
            )
            self._categories[template_cat.name] = category
        return category

    def _register_mpl_cmaps(self):
        for cm_name, cmap in matplotlib.colormaps.items():
            if cm_name.startswith(OCEAN_PREFIX):
                # We have a separate category for OCEAN maps
                continue
            if cm_name.endswith(REVERSED_SUFFIX):
                # We register reversed maps, if any, separately
                continue
            template_cat = _CM_NAME_TO_TEMPLATE_CATEGORY.get(cm_name)
            if template_cat is None:
                template_cat = MISCELLANEOUS_CATEGORY
            self.register_colormap(
                Colormap(
                    cm_name,
                    cat_name=template_cat.name,
                    cmap=cmap,
                    cmap_reversed=matplotlib.colormaps.get(cm_name + REVERSED_SUFFIX),
                )
            )

    def _register_ocm_cmaps(self):
        if cmo is None:
            return
        for cm_name in cmo.__dict__.keys():
            if not isinstance(
                getattr(cmo, cm_name), matplotlib.colors.Colormap
            ) or cm_name.endswith(REVERSED_SUFFIX):
                continue
            self.register_colormap(
                Colormap(
                    cm_name,
                    cat_name=OCEAN_CATEGORY.name,
                    cmap=getattr(cmo, cm_name),
                    cmap_reversed=getattr(cmo, cm_name + REVERSED_SUFFIX, None),
                )
            )


def parse_cm_code(cm_code: str) -> tuple[str, Optional[Colormap]]:
    # Note, if we get performance issues here, we should
    # cache cm_code -> colormap
    values: Optional[list[Union[int, float]]] = None
    try:
        user_color_map: dict[str, Any] = json.loads(cm_code)
        cm_name = user_color_map["name"]
        cm_items = user_color_map["colors"]
        cm_type = user_color_map.get("type", "node")
        cm_base_name, _, _ = parse_cm_name(cm_name)
        n = len(cm_items)
        if cm_type == "key":
            values: list[int] = []
            colors: list[Union[str, tuple[float, ...]]] = []
            bad = 0, 0, 0, 0
            for i, (value, color) in enumerate(cm_items):
                key = int(value)
                values.append(key)
                colors.append(color)
                if i == n - 1:
                    # the last key's next boundary is key+1
                    values.append(key + 1)
                elif key + 1 != int(cm_items[i + 1][0]):
                    # insert transparent region from key+1 to next key
                    values.append(key + 1)
                    colors.append(bad)
            cmap = matplotlib.colors.ListedColormap(colors, name=cm_base_name)
            cmap.set_extremes(bad=bad, under=bad, over=bad)
            print(">>>>>>>>>>>> categorical cmap", cmap)
        else:  # cm_type == "bound" or cm_type == "node"
            values, colors = zip(*cm_items)
            vmin = cm_items[0][0]
            vmax = cm_items[-1][0]
            if vmin != 0 or vmax != 1:
                # Normalize values of cm_items between 0 and 1
                norm_values = (np.array(values) - vmin) / (vmax - vmin)
                cm_items = list(zip(norm_values, colors))
            if cm_type == "bound":
                # Turn cm_items into discrete step function
                stepwise_cm_items = []
                for i, (value, color) in enumerate(cm_items[0:-1]):
                    next_value = cm_items[i + 1][0]
                    stepwise_cm_items.append((value, color))
                    stepwise_cm_items.append((next_value, color))
                cm_items = stepwise_cm_items
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                cm_base_name, cm_items
            )
        return cm_name, Colormap(
            cm_base_name,
            cm_type=cm_type,
            cat_name=CUSTOM_CATEGORY.name,
            cmap=cmap,
            values=values,
        )
    except (SyntaxError, KeyError, ValueError, TypeError):
        # If we arrive here, the submitted user-specific cm_code is wrong
        # We do not log or emit a warning here because this would
        # impact performance for current users of xcube-server.
        # Use fallback color map "Reds" to indicate error
        return "Reds", None


def parse_cm_name(cm_name) -> tuple[str, bool, bool]:
    assert_given(cm_name, name="cm_name")
    alpha = cm_name.endswith(ALPHA_SUFFIX)
    if alpha:
        cm_name = cm_name[0 : -len(ALPHA_SUFFIX)]
    reverse = cm_name.endswith(REVERSED_SUFFIX)
    if reverse:
        cm_name = cm_name[0 : -len(REVERSED_SUFFIX)]
    return cm_name, reverse, alpha


def get_reverse_cmap(
    cm_name: str, cmap: matplotlib.colors.Colormap
) -> tuple[str, matplotlib.colors.Colormap]:
    new_cm_name = cm_name + REVERSED_SUFFIX
    return new_cm_name, cmap.reversed(name=new_cm_name)


def get_alpha_cmap(
    cm_name: str, cmap: matplotlib.colors.Colormap
) -> tuple[str, matplotlib.colors.Colormap]:
    """Add extra colormaps with alpha gradient.
    See http://matplotlib.org/api/colors_api.html
    """
    new_cm_name = cm_name + ALPHA_SUFFIX

    # noinspection SpellCheckingInspection
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap) and hasattr(
        cmap, "_segmentdata"
    ):
        # noinspection PyProtectedMember
        new_segment_data = dict(cmap._segmentdata)
        # let alpha increase from 0.0 to 0.5
        new_segment_data["alpha"] = ((0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0))
        return new_cm_name, matplotlib.colors.LinearSegmentedColormap(
            new_cm_name, new_segment_data
        )

    if isinstance(cmap, matplotlib.colors.ListedColormap):
        new_colors = list(cmap.colors)
        a_slope = 2.0 / cmap.N
        a = 0
        for i in range(len(new_colors)):
            # noinspection PyUnresolvedReferences
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
        return new_cm_name, matplotlib.colors.ListedColormap(
            new_colors, name=new_cm_name
        )

    LOG.warning(f"Could not create alpha colormap for cmap of type {type(cmap)}")

    return cm_name, cmap


def get_cmap_png_base64(cmap: matplotlib.colors.Colormap) -> str:
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient,))
    image_data = cmap(gradient, bytes=True)
    image = Image.fromarray(image_data, "RGBA")

    # For testing, do:
    # stream = io.FileIO('../cmaps/' + cmap_name + '.png', 'wb')
    # image.save(stream, format='PNG')
    # stream.close()

    stream = io.BytesIO()
    image.save(stream, format="PNG")
    cbar_png_bytes = stream.getvalue()
    stream.close()

    cbar_png_data = base64.b64encode(cbar_png_bytes)
    cbar_png_bytes = cbar_png_data.decode("unicode_escape")

    return cbar_png_bytes


def load_custom_colormap(custom_colormap_path: str) -> Optional[Colormap]:
    try:
        # Currently, we only support SNAP *.cpd files
        colormap = load_snap_cpd_colormap(custom_colormap_path)
        LOG.info(f"Loaded custom colormap" f" {custom_colormap_path!r}")
        return colormap
    except FileNotFoundError:
        LOG.error(f"Missing custom colormap file:" f" {custom_colormap_path}")
    except ValueError as e:
        LOG.error(
            f"Detected invalid custom colormap" f" {custom_colormap_path!r}: {e}",
            exc_info=True,
        )
    return None


def load_snap_cpd_colormap(snap_cpd_path: str) -> Colormap:
    points, log_scaled = _parse_snap_cpd_file(snap_cpd_path)
    samples = [sample for sample, _ in points]
    vmin, vmax = min(samples), max(samples)
    if log_scaled:
        norm = matplotlib.colors.LogNorm(vmin, vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin, vmax)
    colors = list(
        zip(map(norm, samples), ["#%02x%02x%02x%02x" % color for _, color in points])
    )
    cm_name, _ = os.path.splitext(os.path.basename(snap_cpd_path))
    return Colormap(
        cm_name,
        cat_name=CUSTOM_CATEGORY.name,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(cm_name, colors),
        norm=norm,
    )


Sample = float
Color = tuple[int, int, int, int]
Palette = list[tuple[Sample, Color]]
LogScaled = bool


def _parse_snap_cpd_file(cpd_file_path: str) -> tuple[Palette, LogScaled]:
    with fsspec.open(cpd_file_path, mode="r") as f:
        illegal_format_msg = f"Illegal SNAP *.cpd format: {cpd_file_path}"

        entries: dict[str, str] = dict()
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
                LOG.warning(
                    f"Unrecognized keyword {keyword!r}"
                    f" in custom colormap {cpd_file_path}"
                )

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
                rgba = tuple(map(int, entries.get(f"color{i}", "").split(",")))
            except ValueError:
                raise ValueError(illegal_format_msg)
            try:
                sample = float(entries.get(f"sample{i}"))
            except ValueError:
                raise ValueError(illegal_format_msg)
            if len(rgba) == 4:
                points.append((sample, rgba))
            elif len(rgba) == 3:
                points.append((sample, (*rgba, 255)))
            else:
                raise ValueError(illegal_format_msg)

        # noinspection PyTypeChecker
        return points, log_scaled
