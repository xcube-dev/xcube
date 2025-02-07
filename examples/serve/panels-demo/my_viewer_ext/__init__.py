#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

from chartlets import Extension

from .my_panel_1 import panel as my_panel_1

# from .my_panel_2 import panel as my_panel_2

ext = Extension(__name__)
ext.add(my_panel_1)
# ext.add(my_panel_2)
