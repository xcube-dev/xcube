# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from chartlets import Extension

from .my_panel_a import panel as my_panel_a
from .my_panel_b import panel as my_panel_b

ext = Extension(__name__)
ext.add(my_panel_a)
ext.add(my_panel_b)
