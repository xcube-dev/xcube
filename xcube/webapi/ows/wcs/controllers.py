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
from datetime import datetime
import numpy as np
import re
import warnings

from typing import Dict, List, Any

from xcube.constants import EXTENSION_POINT_DATASET_IOS
from xcube.util.plugin import get_extension_registry
from xcube.webapi.ows.wcs.context import WcsContext
from xcube.webapi.ows.wmts.controllers import get_crs84_bbox
from xcube.webapi.xml import Document
from xcube.webapi.xml import Element

WCS_VERSION = '1.0.0'
VALID_CRS_LIST = ['EPSG:4326', 'EPSG:3857']

CoverageRequest = Dict[str, Any]


def get_capabilities_xml(ctx: WcsContext, base_url: str) -> str:
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


def get_describe_xml(ctx: WcsContext, coverages: List[str] = None) -> str:
    element = _get_describe_element(ctx, coverages)
    document = Document(element)
    return document.to_xml(indent=4)


def validate_coverage_req(req: CoverageRequest, ctx: WcsContext):
    if 'COVERAGE' in req and is_valid_coverage(req['COVERAGE'], ctx) \
            and 'CRS' in req and is_valid_crs(req['CRS']) \
            and (('BBOX' in req and is_valid_bbox(req['BBOX'])) or
                 ('TIME' in req and is_valid_time(req['TIME']))) \
            and (is_valid_bbox(req['BBOX']) if 'BBOX' in req else True) \
            and (is_valid_time(req['TIME']) if 'TIME' in req else True) \
            and (('WIDTH' in req and 'HEIGHT' in req) or
                 ('RESX' in req and 'RESY' in req)) \
            and (('WIDTH' in req and 'RESX' not in req) or
                 ('WIDTH' in req and 'RESY' not in req) or
                 ('HEIGHT' in req and 'RESX' not in req) or
                 ('HEIGHT' in req and 'RESY' not in req) or
                 ('RESX' in req and 'WIDTH' not in req) or
                 ('RESX' in req and 'HEIGHT' not in req) or
                 ('RESY' in req and 'WIDTH' not in req) or
                 ('RESY' in req and 'HEIGHT' not in req)) \
            and 'FORMAT' in req and is_valid_format(req['FORMAT']) \
            and 'PARAMETER' not in req \
            and 'INTERPOLATION' not in req \
            and 'EXCEPTIONS' not in req:
        return
    elif 'COVERAGE' not in req or not is_valid_coverage(req['COVERAGE'], ctx):
        raise ValueError('No valid value for parameter COVERAGE provided. '
                         'COVERAGE must be a variable name prefixed with '
                         'its dataset name. Example: my_dataset.my_var')
    elif 'PARAMETER' in req:
        raise ValueError('PARAMETER not yet supported')
    elif 'INTERPOLATION' in req:
        raise ValueError('INTERPOLATION not yet supported')
    elif 'EXCEPTIONS' in req:
        raise ValueError('EXCEPTIONS not yet supported')
    elif (('WIDTH' in req and 'HEIGHT' not in req) or
          ('HEIGHT' in req and 'WIDTH' not in req) or
          ('RESX' in req and 'RESY' not in req) or
          ('RESY' in req and 'RESX' not in req) or
          ('WIDTH' in req and 'RESX' in req or 'RESY' in req) or
          ('HEIGHT' in req and 'RESX' in req or 'RESY' in req)):
        raise ValueError('Either both WIDTH and HEIGHT, or both RESX and RESY '
                         'must be provided.')
    elif 'FORMAT' not in req or not is_valid_format(req['FORMAT']):
        raise ValueError('FORMAT wrong or missing. Must be one of ' +
                         ', '.join(_get_formats_list()))
    elif True:
        raise ValueError('Reason unclear, fix me')


def is_valid_coverage(coverage: str, ctx: WcsContext) -> bool:
    band_infos = _extract_band_infos(ctx, [coverage])
    if band_infos:
        return True
    return False


def is_valid_crs(crs: str) -> bool:
    return crs in VALID_CRS_LIST


def is_valid_bbox(bbox: str) -> bool:
    bbox_regex = re.compile(r'-?\d{1,3} -?\d{1,2} -?\d{1,3} -?\d{1,2}')
    if not bbox_regex.match(bbox):
        raise ValueError('BBOX must be given as `minx miny maxx maxy`')
    return True


def is_valid_format(format_req: str) -> bool:
    return format_req in _get_formats_list()


def is_valid_time(time: str) -> bool:
    try:
        datetime.fromisoformat(time)
    except ValueError:
        raise ValueError('TIME value must be given in the format'
                         '\'YYYY-MM-DD[*HH[:MM[:SS[.mmm[mmm]]]]'
                         '[+HH:MM[:SS[.ffffff]]]]\'')
    return True


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
    service_provider = ctx.config.get('ServiceProvider')

    def _get_value(path):
        v = None
        node = service_provider
        for k in path:
            if not isinstance(node, dict) or k not in node:
                return ''
            v = node[k]
            node = v
        return str(v) if v is not None else ''

    def _get_individual_name():
        individual_name = _get_value(['ServiceContact', 'IndividualName'])
        individual_name = tuple(individual_name.split(' ').__reversed__())
        return '{}, {}'.format(*individual_name)

    element = Element('Service', elements=[
        Element('description',
                text=_get_value(['WCS-description'])),
        Element('name',
                text=_get_value(['WCS-name'])),
        Element('label',
                text=_get_value(['WCS-label'])),
        Element('keywords', elements=[
            Element('keyword', text=k) for k in service_provider['keywords']
        ]),
        Element('responsibleParty', elements=[
            Element('individualName',
                    text=_get_individual_name()),
            Element('organisationName',
                    text=_get_value(['ProviderName'])),
            Element('positionName',
                    text=_get_value(['ServiceContact',
                                     'PositionName'])),
            Element('contactInfo', elements=[
                Element('phone', elements=[
                    Element('voice',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Phone',
                                             'Voice'])),
                    Element('facsimile',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Phone',
                                             'Facsimile'])),
                ]),
                Element('address', elements=[
                    Element('deliveryPoint',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'DeliveryPoint'])),
                    Element('city',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'City'])),
                    Element('administrativeArea',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'AdministrativeArea'])),
                    Element('postalCode',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'PostalCode'])),
                    Element('country',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'Country'])),
                    Element('electronicMailAddress',
                            text=_get_value(['ServiceContact',
                                             'ContactInfo',
                                             'Address',
                                             'ElectronicMailAddress'])),
                ]),
                Element('onlineResource', attrs={
                    'xlink:href': _get_value(['ProviderSite'])})
            ]),
        ]),
        Element('fees', text='NONE'),
        Element('accessConstraints', text='NONE')
    ])
    return element


def _get_capability_element(base_url: str) -> Element:
    get_capabilities_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
                           f'request=GetCapabilities'
    describe_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
                   f'request=DescribeCoverage'
    get_url = f'{base_url}?service=WCS&amp;version=1.0.0&amp;' \
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


# noinspection HttpUrlsUsage
def _get_describe_element(ctx: WcsContext, coverages: List[str] = None) \
        -> Element:
    coverage_elements = []

    band_infos = _extract_band_infos(ctx, coverages, True)
    for var_name in band_infos.keys():
        coverage_elements.append(Element('CoverageOffering', elements=[
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
                    Element('gml:Envelope', elements=[
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[0]} '
                                     f'{band_infos[var_name].bbox[1]}'),
                        Element('gml:pos',
                                text=f'{band_infos[var_name].bbox[2]} '
                                     f'{band_infos[var_name].bbox[3]}')
                    ])
                ])
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
                                Element('interval', elements=[
                                    Element('min', text=
                                    f'{band_infos[var_name].min:0.4f}'),
                                    Element('max', text=
                                    f'{band_infos[var_name].max:0.4f}')
                                ])
                            ]),
                        ])
                    ])
                ])
            ]),
            Element('supportedCRSs', elements=[
                Element('requestResponseCRSs', text=' '.join(VALID_CRS_LIST))
            ]),
            Element('supportedFormats', elements=[
                Element('formats', text=f) for f in _get_formats_list()
            ])
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
    formats = get_extension_registry().find_extensions(
        EXTENSION_POINT_DATASET_IOS,
        lambda e: 'w' in e.metadata.get('modes', set())
    )
    return [ext.name for ext in formats if not ext.name == 'mem']


class BandInfo:

    def __init__(self, var_name: str, label: str,
                 bbox: tuple[float, float, float, float]):
        self.var_name = var_name
        self.label = label
        self.bbox = bbox
        self.min = np.nan
        self.max = np.nan


def _extract_band_infos(ctx: WcsContext, coverages: List[str] = None,
                        full: bool = False) -> Dict[str, BandInfo]:
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
            is_spatial_var = var.ndim >= 2 \
                             and var.dims[-1] == x_name \
                             and var.dims[-2] == y_name
            if not is_spatial_var:
                continue

            band_info = BandInfo(qualified_var_name, label, bbox)
            if full:
                nn_values = var.values[~np.isnan(var.values)]
                band_info.min = nn_values.min()
                band_info.max = nn_values.max()

            band_infos[f'{ds_name}.{var_name}'] = band_info

    return band_infos
