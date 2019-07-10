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
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import numpy as np
from PIL import Image

def _get_custom_colormap(color_txt):

    colors = _get_color(color_txt)
    values = _get_tick_val_col(color_txt)

    norm = plt.Normalize(min(values), max(values))
    tuples = list(zip(map(norm, values), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(color_txt, tuples)

    return cmap


def _get_color(color_txt):
    f = open(color_txt, "r")
    lines = f.readlines()
    c = []
    for x in lines:
        if "color" in x:
            r, g, b = (((re.split('\W+', x, 1)[1:])[0].strip()).split(','))
            hex_col = ('#%02x%02x%02x' % (int(r), int(g), int(b)))
            c.append(hex_col)
    f.close()
    return c


def _get_tick_val_col(color_txt):
    f = open(color_txt, "r")
    lines = f.readlines()
    values = []
    for x in lines:
        if "sample" in x:
            value = ((re.split('\W+', x, 1)[1:])[0].strip())
            values.append(float(value))
    f.close()
    return values


def get_norm(color_txt):

    values = _get_tick_val_col(color_txt)
    norm = matplotlib.colors.LogNorm(min(values), max(values))

    return norm, values