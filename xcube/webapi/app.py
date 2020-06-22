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

import os

from tornado.web import Application, StaticFileHandler

from xcube.webapi.context import normalize_prefix
from xcube.webapi.handlers import GetNE2TileHandler, GetDatasetVarTileHandler, InfoHandler, GetNE2TileGridHandler, \
    GetDatasetVarTileGridHandler, GetWMTSCapabilitiesXmlHandler, GetColorBarsJsonHandler, GetColorBarsHtmlHandler, \
    GetDatasetsHandler, FindPlacesHandler, FindDatasetPlacesHandler, \
    GetDatasetCoordsHandler, GetTimeSeriesHandler, GetTsLegacyInfoHandler, GetTsLegacyForPointHandler, WMTSKvpHandler, \
    GetTsLegacyForGeometryHandler, GetTsLegacyForFeaturesHandler, GetTsLegacyForGeometriesHandler, \
    GetPlaceGroupsHandler, GetDatasetVarLegendHandler, GetDatasetHandler, GetWMTSTileHandler, GetS3BucketObjectHandler, \
    ListS3BucketHandler, GetDatasetPlaceGroupsHandler, GetDatasetPlaceGroupHandler
from xcube.webapi.service import url_pattern

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


def new_application(route_prefix: str = None, base_dir: str = '.'):
    route_prefix = normalize_prefix(route_prefix)

    application = Application([
        (route_prefix + '/res/(.*)',
         StaticFileHandler, {'path': os.path.join(os.path.dirname(__file__), 'res')}),

        (route_prefix + '/images/(.*)',
         StaticFileHandler, {'path': os.path.join(base_dir, 'images')}),

        (route_prefix + url_pattern('/'),
         InfoHandler),

        (route_prefix + url_pattern('/wmts/1.0.0/WMTSCapabilities.xml'),
         GetWMTSCapabilitiesXmlHandler),
        (route_prefix + url_pattern('/wmts/1.0.0/tile/{{ds_id}}/{{var_name}}/{{z}}/{{y}}/{{x}}.png'),
         GetWMTSTileHandler),
        (route_prefix + url_pattern('/wmts/kvp'),
         WMTSKvpHandler),

        (route_prefix + url_pattern('/datasets'),
         GetDatasetsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}'),
         GetDatasetHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/places'),
         GetDatasetPlaceGroupsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/places/{{place_group_id}}'),
         GetDatasetPlaceGroupHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/coords/{{dim_name}}'),
         GetDatasetCoordsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/legend.png'),
         GetDatasetVarLegendHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/tiles/{{z}}/{{x}}/{{y}}.png'),
         GetDatasetVarTileHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/tilegrid'),
         GetDatasetVarTileGridHandler),

        # AWS S3 compatible data access as ZARR

        (route_prefix + url_pattern('/s3bucket/{{ds_id}}/(?P<path>.*)'),
         GetS3BucketObjectHandler),
        (route_prefix + url_pattern('/s3bucket/{{ds_id}}'),
         GetS3BucketObjectHandler),
        (route_prefix + url_pattern('/s3bucket'),
         ListS3BucketHandler),

        # Natural Earth 2 tiles for testing

        (route_prefix + url_pattern('/ne2/tilegrid'),
         GetNE2TileGridHandler),
        (route_prefix + url_pattern('/ne2/tiles/{{z}}/{{x}}/{{y}}.jpg'),
         GetNE2TileHandler),

        # Color Bars API

        (route_prefix + url_pattern('/colorbars'),
         GetColorBarsJsonHandler),
        (route_prefix + url_pattern('/colorbars.html'),
         GetColorBarsHtmlHandler),

        # Places API (PRELIMINARY & UNSTABLE - will be revised soon)

        (route_prefix + url_pattern('/places'),
         GetPlaceGroupsHandler),
        (route_prefix + url_pattern('/places/{{place_group_id}}'),
         FindPlacesHandler),
        (route_prefix + url_pattern('/places/{{place_group_id}}/{{ds_id}}'),
         FindDatasetPlacesHandler),

        # Time-Series API

        (route_prefix + url_pattern('/timeseries/{{ds_id}}/{{var_name}}'),
         GetTimeSeriesHandler),

        # Legacy time-series API (for VITO's DCS4COP viewer only)

        (route_prefix + url_pattern('/ts'),
         GetTsLegacyInfoHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/point'),
         GetTsLegacyForPointHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/geometry'),
         GetTsLegacyForGeometryHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/geometries'),
         GetTsLegacyForGeometriesHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/features'),
         GetTsLegacyForFeaturesHandler),
    ])
    return application
