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

from xcube.webapi.ows.wcs.context import WcsContext
from xcube.webapi.ows.wmts.controllers import get_crs84_bbox

from xcube.webapi.xml import Document
from xcube.webapi.xml import Element

WCS_VERSION = '1.0.0'


def get_wcs_capabilities_xml(ctx: WcsContext, base_url: str) -> str:
    """
    Get WCSCapabilities.xml according to
    https://schemas.opengis.net/wcs/1.0.0/.

    :param ctx: server context
    :param base_url: the request base URL
    :return: XML plain text in UTF-8 encoding
    """
    element = get_capabilities_element(ctx, base_url)
    document = Document(element)
    return document.to_xml(indent=4)


# noinspection HttpUrlsUsage
def get_capabilities_element(ctx: WcsContext,
                             base_url: str) -> Element:
    service_element = get_service_element(ctx)
    capability_element = get_capability_element(base_url)
    content_element = Element('ContentMetadata')

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
            var = ds[var_name]

            label = var.long_name if hasattr(var, 'long_name') else var_name
            is_spatial_var = var.ndim >= 2 \
                             and var.dims[-1] == x_name \
                             and var.dims[-2] == y_name
            if not is_spatial_var:
                continue

            content_element.add(Element('CoverageOfferingBrief', elements=[
                Element('name', text=var_name),
                Element('label', text=label),
                Element('lonLatEnvelope', elements=[
                    Element('gml:pos', text=f'{bbox[0]}'
                                            f' {bbox[1]}'),
                    Element('gml:pos', text=f'{bbox[2]}'
                                            f' {bbox[3]}')
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


def get_service_element(ctx: WcsContext) -> Element:
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


def get_capability_element(base_url: str) -> Element:
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
