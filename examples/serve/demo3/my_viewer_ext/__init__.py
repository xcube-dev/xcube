from dashipy import Extension
from .my_panel_1 import panel as my_panel_1

ext = Extension(__name__)
ext.add(my_panel_1)