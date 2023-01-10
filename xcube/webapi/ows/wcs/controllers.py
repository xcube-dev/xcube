# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
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

import warnings
from typing import Dict, List, Any, Union

import numpy as np
import xarray as xr

from xcube.core.tile import BBox
from xcube.core.tile import compute_tiles
from xcube.server.api import ApiError
from xcube.webapi.common.xml import Document
from xcube.webapi.common.xml import Element
from xcube.webapi.datasets.context import DatasetConfig
from xcube.webapi.ows.wcs.context import WcsContext
from xcube.webapi.ows.wmts.controllers import get_crs84_bbox

WCS_VERSION = '1.0.0'
VALID_CRS_LIST = ['EPSG:4326']


# VALID_CRS_LIST = ['EPSG:4326', 'EPSG:3857']


class CoverageRequest:
    coverage = None
    crs = None
    bbox = None
    time = None
    width = None
    height = None
    format = None
    resx = None
    resy = None
    interpolation = None
    parameter = None
    exceptions = None

    def __init__(self, req: Dict[str, Any]):
        if 'COVERAGE' in req:
            self.coverage = req['COVERAGE']
        if 'CRS' in req:
            self.crs = req['CRS']
        if 'BBOX' in req:
            self.bbox = req['BBOX']
        if 'TIME' in req:
            self.time = req['TIME']
        if 'WIDTH' in req:
            self.width = req['WIDTH']
        if 'HEIGHT' in req:
            self.height = req['HEIGHT']
        if 'FORMAT' in req:
            self.format = req['FORMAT']
        if 'RESX' in req:
            self.resx = req['RESX']
        if 'RESY' in req:
            self.resy = req['RESY']
        if 'INTERPOLATION' in req:
            self.interpolation = req['INTERPOLATION']
        if 'PARAMETER' in req:
            self.parameter = req['PARAMETER']
        if 'EXCEPTIONS' in req:
            self.exceptions = req['EXCEPTIONS']


def get_wcs_capabilities_xml(ctx: WcsContext, base_url: str) -> str:
    """
    Get WCSCapabilities.xml according to
    https://schemas.opengis.net/wcs/1.0.0/.

    :param ctx: server context
    :param base_url: the request base URL
    :return: XML plain text in UTF-8 encoding
    """
    element = _get_capabilities_element(ctx, base_url)
    document = Document(element)
    return document.to_xml(indent=4)


def get_describe_coverage_xml(ctx: WcsContext,
                              coverages: List[str] = None) -> str:
    element = _get_describe_element(ctx, coverages)
    document = Document(element)
    return document.to_xml(indent=4)


def get_coverage(ctx: WcsContext, req: CoverageRequest) -> xr.Dataset:
    dataset_config = _get_dataset_config(ctx, req)
    ds_name = dataset_config['Identifier']
    ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_name)

    # TODO (forman): compute optimal level for RES_X/RES_Y

    bbox: Union[BBox, None]
    try:
        # noinspection PyTypeChecker
        bbox = tuple(float(v) for v in req.bbox.split(','))
    except ValueError:
        bbox = None
    if not bbox or len(bbox) != 4:
        raise ApiError.BadRequest('invalid bbox')

    ds_name = dataset_config['Identifier']
    var_name = req.coverage.replace(ds_name + '.', '')
    tile_size = int(req.width), int(req.height)
    dataset = compute_tiles(ml_dataset, var_name, bbox, req.crs,
                            tile_size=tile_size, as_dataset=True)
    dataset = dataset.rename_dims({'x': 'lon', 'y': 'lat'})
    dataset = dataset.rename_vars({'x': 'lon', 'y': 'lat'})
    return dataset


def _get_dataset_config(ctx: WcsContext, req: CoverageRequest) \
        -> DatasetConfig:
    # TODO: too much computation here, precompute mapping and store in
    #   WCS context object, so the dataset config can be looked up:
    #
    #   class WcsContext(...):
    #     ...
    #     def get_dataset_config(self, coverage: str):
    #        ds_config = self.coverage_to_ds_config.get(coverage)
    #        if ds_config is None:
    #           raise ApiError.BadRequest(f'unknown coverage {coverage!r}')
    #        return ds_config
    #
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ds = ctx.datasets_ctx.get_dataset(ds_name)

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if req.coverage == qualified_var_name:
                return dataset_config
    raise RuntimeError('Should never come here. Contact the developers.')


# noinspection HttpUrlsUsage
def _get_capabilities_element(ctx: WcsContext,
                              base_url: str) -> Element:
    service_element = _get_service_element(ctx)
    capability_element = _get_capability_element(base_url)
    content_element = Element('ContentMetadata')

    band_infos = _extract_band_infos(ctx)
    for var_name in band_infos.keys():
        content_element.add(Element('CoverageOfferingBrief', elements=[
            Element('name', text=var_name),
            Element('label', text=band_infos[var_name].label),
            Element('lonLatEnvelope', elements=[
                Element('gml:pos', text=f'{band_infos[var_name].bbox[0]}'
                                        f' {band_infos[var_name].bbox[1]}'),
                Element('gml:pos', text=f'{band_infos[var_name].bbox[2]}'
                                        f' {band_infos[var_name].bbox[3]}')
            ])
        ]))

    return Element(
        'WCS_Capabilities',
        attrs={
            'xmlns': "http://www.opengis.net/wcs",
            'xmlns:gml': "http://www.opengis.net/gml",
            'xmlns:xlink': "http://www.w3.org/1999/xlink",
            'version': WCS_VERSION,
        },
        elements=[
            service_element,
            capability_element,
            content_element
        ]
    )


def _get_service_element(ctx: WcsContext) -> Element:
    service_provider = ctx.config.get('ServiceProvider', {})
    wcs_metadata = ctx.config.get('WebCoverageService', {})

    def _get_sp_value(path):
        v = None
        node = service_provider
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return ''
            v = node[k]
            node = v
        return str(v) if v is not None else ''

    def _get_individual_name() -> str:
        individual_name = _get_sp_value(['ServiceContact', 'IndividualName'])
        if not individual_name:
            return ''
        individual_name = tuple(individual_name.split(' ').__reversed__())
        return '{}, {}'.format(*individual_name)

    element = Element('Service', elements=[
        Element('description',
                text=wcs_metadata.get('Description',
                                      'xcube WCS 1.0 API')),
        Element('name',
                text=wcs_metadata.get('Name', 'xcube WCS')),
        Element('label',
                text=wcs_metadata.get('Label', 'xcube WCS')),
        Element('keywords', elements=[
            Element('keyword', text=k) for k in wcs_metadata.get('Keywords',
                                                                 [])
        ]),
        Element('responsibleParty', elements=[
            Element('individualName',
                    text=_get_individual_name()),
            Element('organisationName',
                    text=_get_sp_value(['ProviderName'])),
            Element('positionName',
                    text=_get_sp_value(['ServiceContact',
                                        'PositionName'])),
            Element('contactInfo', elements=[
                Element('phone', elements=[
                    Element('voice',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Phone',
                                                'Voice'])),
                    Element('facsimile',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Phone',
                                                'Facsimile'])),
                ]),
                Element('address', elements=[
                    Element('deliveryPoint',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'DeliveryPoint'])),
                    Element('city',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'City'])),
                    Element('administrativeArea',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'AdministrativeArea'])),
                    Element('postalCode',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'PostalCode'])),
                    Element('country',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'Country'])),
                    Element('electronicMailAddress',
                            text=_get_sp_value(['ServiceContact',
                                                'ContactInfo',
                                                'Address',
                                                'ElectronicMailAddress'])),
                ]),
                Element('onlineResource', attrs={
                    'xlink:href': _get_sp_value(['ProviderSite'])})
            ]),
        ]),
        Element('fees', text='NONE'),
        Element('accessConstraints', text='NONE')
    ])
    return element


def _get_capability_element(base_url: str) -> Element:
    get_capabilities_url = f'{base_url}/wcs/kvp?service=WCS&amp;' \
                           f'version=1.0.0&amp;request=GetCapabilities'
    describe_url = f'{base_url}/wcs/kvp?service=WCS&amp;version=1.0.0&amp;' \
                   f'request=DescribeCoverage'
    get_url = f'{base_url}/wcs/kvp?service=WCS&amp;version=1.0.0&amp;' \
              f'request=GetCoverage'
    return Element('Capability', elements=[
        Element('Request', elements=[
            Element('GetCapabilities', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': get_capabilities_url})
                        ])
                    ])
                ])
            ]),
            Element('DescribeCoverage', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': describe_url})
                        ])
                    ])
                ])
            ]),
            Element('GetCoverage', elements=[
                Element('DCPType', elements=[
                    Element('HTTP', elements=[
                        Element('Get', elements=[
                            Element('OnlineResource',
                                    attrs={'xlink:href': get_url})
                        ])
                    ])
                ])
            ]),
        ]),
        Element('Exception', elements=[
            Element('Format', text='application/x-ogc-wcs')
        ])
    ])


def _get_describe_element(ctx: WcsContext, coverages: List[str] = None) \
        -> Element:
    coverage_elements = []

    band_infos = _extract_band_infos(ctx, coverages)
    for var_name in band_infos.keys():
        coverage_elements.append(Element('CoverageOffering', elements=[
            Element('description', text=band_infos[var_name].label),
            Element('name', text=var_name),
            Element('label', text=band_infos[var_name].label),
            Element('lonLatEnvelope', elements=[
                Element('gml:pos', text=f'{band_infos[var_name].bbox[0]} '
                                        f'{band_infos[var_name].bbox[1]}'),
                Element('gml:pos', text=f'{band_infos[var_name].bbox[2]} '
                                        f'{band_infos[var_name].bbox[3]}')
            ]),
            Element('domainSet', elements=[
                Element('spatialDomain', elements=[
                    Element('gml:Envelope', attrs={'srsName': 'EPSG:4326'},
                            elements=[
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[0]} '
                                     f'{band_infos[var_name].bbox[1]}'),
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[2]} '
                                     f'{band_infos[var_name].bbox[3]}')
                    ])
                ]),
                Element('temporalDomain', elements=[
                    Element('gml:timePosition', text=time_step)
                    for time_step in band_infos[var_name].time_steps])
            ]),
            Element('rangeSet', elements=[
                Element('RangeSet', elements=[
                    Element('name', text=var_name),
                    Element('label', text=band_infos[var_name].label),
                    Element('axisDescription', elements=[
                        Element('AxisDescription', elements=[
                            Element('name', text='Band'),
                            Element('label', text='Band'),
                            Element('values', elements=[
                                Element('singleValue', text='1')
                            ])
                        ])
                    ])
                ])
            ]),
            Element('supportedCRSs', elements=[
                Element('requestResponseCRSs', text='EPSG:4326')
                # todo - find out why this does not work with qgis
                # Element('requestResponseCRSs',
                #         text=','.join(VALID_CRS_LIST))
            ]),
            Element('supportedFormats', elements=[
                Element('formats', text=f) for f in _get_formats_list()
            ]),
            Element('supportedInterpolations',
                    attrs=dict(default='nearest neighbor'),
                    elements=[
                        # Respect BBOX only
                        Element('interpolationMethod',
                                text='none'),
                        # Respect BBOX and WIDTH,HEIGHT or RESX,RESY
                        Element('interpolationMethod',
                                text='nearest neighbor'),
                    ]),
        ]))

    return Element(
        'CoverageDescription',
        attrs={
            'xmlns': "http://www.opengis.net/wcs",
            'xmlns:gml': "http://www.opengis.net/gml",
            'version': WCS_VERSION,
        },
        elements=coverage_elements
    )


def _get_formats_list() -> List[str]:
    # We currently only support NetCDF, because
    # 1. QGIS understands them
    # 2. response can be a single file
    return ['netcdf']


class BandInfo:

    def __init__(self, var_name: str, label: str,
                 bbox: tuple[float, float, float, float],
                 time_steps: list[str]):
        self.var_name = var_name
        self.label = label
        self.bbox = bbox
        self.min = np.nan
        self.max = np.nan
        self.time_steps = time_steps


def _extract_band_infos(ctx: WcsContext, coverages: List[str] = None) \
        -> Dict[str, BandInfo]:
    band_infos = {}
    for dataset_config in ctx.datasets_ctx.get_dataset_configs():
        ds_name = dataset_config['Identifier']
        ml_dataset = ctx.datasets_ctx.get_ml_dataset(ds_name)
        grid_mapping = ml_dataset.grid_mapping
        ds = ml_dataset.base_dataset

        try:
            bbox = get_crs84_bbox(grid_mapping)
        except ValueError:
            warnings.warn(f'cannot compute geographical'
                          f' bounds for dataset {ds_name}, ignoring it')
            continue

        x_name, y_name = grid_mapping.xy_dim_names

        var_names = sorted(ds.data_vars)
        for var_name in var_names:
            qualified_var_name = f'{ds_name}.{var_name}'
            if coverages and qualified_var_name not in coverages:
                continue
            var = ds[var_name]

            label = var.long_name if hasattr(var, 'long_name') else var_name
            label += f' (from {ds_name})'
            is_spatial_var = var.ndim >= 2 \
                             and var.dims[-1] == x_name \
                             and var.dims[-2] == y_name
            if not is_spatial_var:
                continue

            is_temporal_var = var.ndim >= 3
            time_steps = None
            if is_temporal_var:
                time_steps = [f'{str(d)[:19]}Z' for d in var.time.values]

            band_info = BandInfo(qualified_var_name, label, bbox, time_steps)
            band_infos[f'{ds_name}.{var_name}'] = band_info

    return band_infos
