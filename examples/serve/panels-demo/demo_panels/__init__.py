#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

from chartlets import Extension

from .panel_histo2d import panel as histo2d_panel
from .panel_spectrum import panel as spectrum_panel
from .panel_demo import panel as demo_panel

ext = Extension(__name__)
ext.add(histo2d_panel)
ext.add(spectrum_panel)
ext.add(demo_panel)
