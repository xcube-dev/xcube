from chartlets import Extension
from .my_panel_a import panel as my_panel_a
from .my_panel_b import panel as my_panel_b

ext = Extension(__name__)
ext.add(my_panel_a)
ext.add(my_panel_b)
