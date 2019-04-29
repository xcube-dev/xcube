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

import os

from . import __version__

DEFAULT_NAME = 'xcube'
DEFAULT_ADDRESS = 'localhost'
DEFAULT_PORT = 8080
DEFAULT_CONFIG_FILE = os.path.abspath('xcube_server.yml')
DEFAULT_TILE_CACHE_SIZE = "512M"
DEFAULT_UPDATE_PERIOD = 2.
DEFAULT_LOG_PREFIX = os.path.abspath('xcube_server.log')
DEFAULT_TILE_COMP_MODE = 0
DEFAULT_TRACE_PERF = False

DEFAULT_CMAP_CBAR = 'jet'
DEFAULT_CMAP_VMIN = 0.
DEFAULT_CMAP_VMAX = 1.
DEFAULT_CMAP_WIDTH = 1
DEFAULT_CMAP_HEIGHT = 5

_GIGAS = 1000 * 1000 * 1000

FILE_TILE_CACHE_CAPACITY = 20 * _GIGAS
FILE_TILE_CACHE_ENABLED = False
FILE_TILE_CACHE_PATH = './image-cache'

MEM_TILE_CACHE_CAPACITY = 2 * _GIGAS

API_PREFIX = f"/api/{__version__}"
