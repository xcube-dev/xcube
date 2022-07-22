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

# TODO (forman): xcube Server NG: remove this module, must no longer be used

import os

from tornado.web import Application, StaticFileHandler

import xcube.webapi.handlers
from xcube.webapi.context import normalize_prefix
from xcube.webapi.service import url_pattern

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


def new_application(route_prefix: str = None, base_dir: str = '.'):
    route_prefix = normalize_prefix(route_prefix)

    application = Application([
        (route_prefix + '/res/(.*)',
         StaticFileHandler, {'path': os.path.join(os.path.dirname(__file__), 'res')}),

        (route_prefix + '/images/(.*)',
         StaticFileHandler, {'path': os.path.join(base_dir, 'images')}),

        # App Info API
        (route_prefix + url_pattern('/'),
         xcube.webapi.handlers.InfoHandler),

        # App Maintenance API
        (route_prefix + url_pattern('/maintenance/{{action}}'),
         xcube.webapi.handlers.MaintenanceHandler),

        # WMTS 1.0 API
        (route_prefix + url_pattern('/wmts/1.0.0/WMTSCapabilities.xml'),
         xcube.webapi.handlers.GetWMTSCapabilitiesXmlHandler),
        (route_prefix + url_pattern('/wmts/1.0.0/{{tms_id}}/WMTSCapabilities.xml'),
         xcube.webapi.handlers.GetWMTSCapabilitiesXmlTmsHandler),
        (route_prefix + url_pattern('/wmts/1.0.0/tile/{{ds_id}}/{{var_name}}/{{z}}/{{y}}/{{x}}.png'),
         xcube.webapi.handlers.GetWMTSTileHandler),
        (route_prefix + url_pattern('/wmts/1.0.0/tile/{{ds_id}}/{{var_name}}/{{tms_id}}/{{z}}/{{y}}/{{x}}.png'),
         xcube.webapi.handlers.GetWMTSTileTmsHandler),
        (route_prefix + url_pattern('/wmts/kvp'),
         xcube.webapi.handlers.WMTSKvpHandler),

        # Datasets API

        (route_prefix + url_pattern('/datasets'),
         xcube.webapi.handlers.GetDatasetsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}'),
         xcube.webapi.handlers.GetDatasetHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/places'),
         xcube.webapi.handlers.GetDatasetPlaceGroupsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/places/{{place_group_id}}'),
         xcube.webapi.handlers.GetDatasetPlaceGroupHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/coords/{{dim_name}}'),
         xcube.webapi.handlers.GetDatasetCoordsHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/legend.png'),
         xcube.webapi.handlers.GetDatasetVarLegendHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/tiles/{{z}}/{{x}}/{{y}}.png'),
         xcube.webapi.handlers.GetDatasetVarTileHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/tilegrid'),
         xcube.webapi.handlers.GetDatasetVarTileGridHandler),
        (route_prefix + url_pattern('/datasets/{{ds_id}}/vars/{{var_name}}/tiles2/{{z}}/{{y}}/{{x}}'),
         xcube.webapi.handlers.GetDatasetVarTile2Handler),

        # AWS S3 compatible data access as ZARR

        (route_prefix + url_pattern('/s3bucket/{{ds_id}}/(?P<path>.*)'),
         xcube.webapi.handlers.GetS3BucketObjectHandler),
        (route_prefix + url_pattern('/s3bucket/{{ds_id}}'),
         xcube.webapi.handlers.GetS3BucketObjectHandler),
        (route_prefix + url_pattern('/s3bucket'),
         xcube.webapi.handlers.ListS3BucketHandler),

        # Color Bars API

        (route_prefix + url_pattern('/colorbars'),
         xcube.webapi.handlers.GetColorBarsJsonHandler),
        (route_prefix + url_pattern('/colorbars.html'),
         xcube.webapi.handlers.GetColorBarsHtmlHandler),

        # Places API (PRELIMINARY & UNSTABLE - will be revised soon)

        (route_prefix + url_pattern('/places'),
         xcube.webapi.handlers.GetPlaceGroupsHandler),
        (route_prefix + url_pattern('/places/{{place_group_id}}'),
         xcube.webapi.handlers.FindPlacesHandler),
        (route_prefix + url_pattern('/places/{{place_group_id}}/{{ds_id}}'),
         xcube.webapi.handlers.FindDatasetPlacesHandler),

        # Time-Series API

        (route_prefix + url_pattern('/timeseries/{{ds_id}}/{{var_name}}'),
         xcube.webapi.handlers.GetTimeSeriesHandler),

        # Legacy time-series API (for VITO's DCS4COP viewer only)

        (route_prefix + url_pattern('/ts'),
         xcube.webapi.handlers.GetTsLegacyInfoHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/point'),
         xcube.webapi.handlers.GetTsLegacyForPointHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/geometry'),
         xcube.webapi.handlers.GetTsLegacyForGeometryHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/geometries'),
         xcube.webapi.handlers.GetTsLegacyForGeometriesHandler),
        (route_prefix + url_pattern('/ts/{{ds_id}}/{{var_name}}/features'),
         xcube.webapi.handlers.GetTsLegacyForFeaturesHandler),
    ])
    return application
