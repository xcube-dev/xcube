# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import tempfile
from typing import Optional

import dask.array
from xarray import Dataset

from xcube.constants import LOG
from xcube.server.api import ApiError
from xcube.server.api import ApiHandler
from .api import api
from .context import WcsContext
from .controllers import CoverageRequest
from .controllers import get_coverage
from .controllers import get_describe_coverage_xml
from .controllers import get_wcs_capabilities_xml

WCS_VERSION = '1.0.0'

# Number of bytes to read and write at once
IO_CHUNK_SIZE = 4 * 1024 * 1024


@api.route('/wcs/1.0.0/WCSCapabilities.xml')
class WcsCapabilitiesXmlHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='getWcsCapabilities',
                   summary='Gets the WCS capabilities as XML document')
    async def get(self):
        self.request.make_query_lower_case()
        capabilities = await self.ctx.run_in_executor(
            None,
            get_wcs_capabilities_xml,
            self.ctx,
            self.request.base_url
        )
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(capabilities)


@api.route('/wcs/kvp')
class WcsKvpHandler(ApiHandler[WcsContext]):
    @api.operation(operationId='invokeWcsMethodFromKvp',
                   summary='Invokes the WCS by key-value pairs')
    async def get(self):
        self.request.make_query_lower_case()
        service = self.request.get_query_arg('service', default='WCS')
        if service != 'WCS':
            raise ApiError.BadRequest(
                'value for "service" parameter must be "WCS"'
            )
        request = self.request.get_query_arg('request')
        if request == "GetCapabilities":
            await self._do_get_capabilities()
        elif request == 'DescribeCoverage':
            await self._do_describe_coverage()
        elif request == "GetCoverage":
            await self._do_get_coverage()
        else:
            raise ApiError.BadRequest(
                f'invalid request type "{request}"'
            )

    async def _do_get_coverage(self):
        wcs_version = self.request.get_query_arg('version',
                                                 default=WCS_VERSION)
        if wcs_version != WCS_VERSION:
            raise ApiError.BadRequest(
                f'value for "version" parameter must be "{WCS_VERSION}"'
            )
        coverage = self.request.get_query_arg('coverage')
        if not coverage:
            raise ApiError.BadRequest(
                'missing query argument "coverage"'
            )
        request_crs = self.request.get_query_arg('crs')
        if not request_crs:
            raise ApiError.BadRequest(
                'missing query argument "crs"'
            )
        response_crs = self.request.get_query_arg('response_crs',
                                                  default=request_crs)
        time = self.request.get_query_arg('time')
        if not time:
            raise ApiError.BadRequest(
                'missing value for query argument "time"'
            )
        # QGIS specific hack!
        time = time.replace('Z', '')
        file_format = self.request.get_query_arg('format',
                                                 default='geotiff')
        file_format = file_format.lower()
        if file_format not in ("geotiff", "netcdf"):
            raise ApiError.BadRequest(
                f'value for "format" not supported: "{file_format}"'
            )
        bbox = self.request.get_query_arg('bbox',
                                          default='-180,90,180,-90')
        width = self.request.get_query_arg('width')
        height = self.request.get_query_arg('height')
        resx = self.request.get_query_arg('resx')
        resy = self.request.get_query_arg('resy')
        cov_req = CoverageRequest({
            'COVERAGE': coverage,
            'CRS': response_crs,
            'TIME': time,
            'BBOX': bbox,
            'FORMAT': file_format,
            'WIDTH': width,
            'HEIGHT': height,
            'RESX': resx,
            'RESY': resy
        })

        try:
            cube = await self.ctx.run_in_executor(
                None,
                get_coverage,
                self.ctx,
                cov_req
            )
        except ValueError as e:
            raise ApiError.BadRequest(f'{e}') from e

        cube = self._clean_cube(cube, time)

        self.response.set_header('Content-Type',
                                 'application/octet-stream')

        if file_format == 'netcdf':
            temp_file_path = await self.ctx.run_in_executor(
                None, self._write_netcdf, cube
            )
        else:
            temp_file_path = await self.ctx.run_in_executor(
                None, self._write_geotiff, cube
            )

        with open(temp_file_path, 'rb') as tf:
            while True:
                chunk = tf.read(IO_CHUNK_SIZE)
                if chunk is None or len(chunk) == 0:
                    break
                try:
                    self.response.write(chunk)
                except (OSError, IOError) as e:
                    raise ApiError.InternalServerError(
                        f'failed writing to {temp_file_path}: {e}'
                    ) from e

            await self.response.finish()

        try:
            os.remove(temp_file_path)
        except (OSError, IOError) as e:
            LOG.error(f'Failed to remove'
                      f' temporary file {temp_file_path}: {e}',
                      exc_info=True)

    async def _do_describe_coverage(self):
        wcs_version = self.request.get_query_arg('version',
                                                 default=WCS_VERSION)
        if wcs_version != WCS_VERSION:
            raise ApiError.BadRequest(
                f'value for "version" parameter must be "{WCS_VERSION}"'
            )
        coverages = self.request.get_query_arg("coverage")
        if coverages:
            coverages = coverages.split(',')
        describe_coverage_xml = await self.ctx.run_in_executor(
            None,
            get_describe_coverage_xml,
            self.ctx,
            coverages
        )
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(describe_coverage_xml)

    async def _do_get_capabilities(self):
        wcs_version = self.request.get_query_arg(
            "version", default=WCS_VERSION
        )
        if wcs_version != WCS_VERSION:
            raise ApiError.BadRequest(
                f'value for "version" parameter must be "{WCS_VERSION}"'
            )
        capabilities_xml = await self.ctx.run_in_executor(
            None,
            get_wcs_capabilities_xml,
            self.ctx,
            self.request.base_url
        )
        self.response.set_header('Content-Type', 'application/xml')
        await self.response.finish(capabilities_xml)

    @staticmethod
    def _write_geotiff(dataset: Dataset) -> str:
        with tempfile.NamedTemporaryFile(prefix='xcube-wcs-',
                                         suffix='.tif',
                                         delete=False) as tf:
            dataset.rio.to_raster(tf.name)
            return tf.name

    @staticmethod
    def _write_netcdf(dataset: Dataset) -> str:
        with tempfile.NamedTemporaryFile(prefix='xcube-wcs-',
                                         suffix='.nc',
                                         delete=False) as tf:
            dataset.to_netcdf(path=tf.name, mode='w')
            return tf.name

    @staticmethod
    def _clean_cube(dataset: Dataset, time: Optional[str] = None) -> Dataset:
        # TODO (forman): FIXME: Cubes generated by gen2
        #   have a non-JSON-serializable value in "history" attribute.
        dataset.attrs.pop('history', None)
        # GeoTIFF doesn't like variables with non-spatial dimensions
        # Here it is the dimension "bnds":
        for var_name in ['lat_bnds', 'lon_bnds', 'time_bnds']:
            if var_name in dataset:
                dataset = dataset.drop_vars(var_name)
        # Select desired time slice
        if 'time' in dataset.coords:
            if time is None:
                dataset = dataset.isel(time=-1)
            else:
                dataset = dataset.sel(time=time, method='nearest')
        return dataset


def _query_to_dict(request):
    return {k: v[0] for k, v in request.query.items()}
