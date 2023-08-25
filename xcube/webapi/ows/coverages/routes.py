# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from xcube.server.api import ApiHandler

from .api import api
from .context import CoveragesContext
from .controllers import get_coverage, get_coverage_domainset, \
    get_coverage_rangetype


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}/coverage")
class CatalogRootHandler(ApiHandler[CoveragesContext]):
    @api.operation(operation_id='coveragesCoverage',
                   summary='A coverage in OGC API - Coverages')
    async def get(self, collectionId: str):
        accept = self.request.headers.get('Accept').split(',')
        # TODO Better content negotiation; there are libraries for this.
        ds = get_coverage(self.ctx.datasets_ctx, collectionId)
        if 'text/html' in accept:
            result = (f'<html><title>Collection</title><body>'
                      f'<p>{collectionId}</p>'
                      f'<p>{ds.ds_id}</p>'
                      f'</body></html>')
        else:
            # TODO: replace this placeholder output with actual coverage data
            result = \
                {
                    'id': collectionId,
                    'type': 'CoverageByDomainAndRange',
                    'envelope': {
                        'type': 'EnvelopeByAxis',
                        'id': 'string',
                        'srsName': 'string',
                        'axisLabels': [
                            'string'
                        ],
                        'axis': [
                            {
                                'type': 'AxisExtent',
                                'id': 'string',
                                'axisLabel': 'string',
                                'lowerBound': 0,
                                'upperBound': 0,
                                'uomLabel': 'string'
                            }
                        ]
                    },
                    'domainSet': {
                        'type': 'DomainSet',
                        'generalGrid': {
                            'type': 'GeneralGridCoverage',
                            'id': 'string',
                            'srsName': 'string',
                            'axisLabels': [
                                'string'
                            ],
                            'axis': [
                                {
                                    'type': 'IndexAxis',
                                    'id': 'string',
                                    'axisLabel': 'string',
                                    'lowerBound': 0,
                                    'upperBound': 0
                                },
                                {
                                    'type': 'RegularAxis',
                                    'id': 'string',
                                    'axisLabel': 'string',
                                    'lowerBound': 'string',
                                    'upperBound': 'string',
                                    'uomLabel': 'string',
                                    'resolution': 0
                                },
                                {
                                    'type': 'IrregularAxis',
                                    'id': 'string',
                                    'axisLabel': 'string',
                                    'uomLabel': 'string',
                                    'coordinate': [
                                        'string'
                                    ]
                                }
                            ],
                            'displacement': {
                                'type': 'DisplacementAxisNest',
                                'id': 'string',
                                'axisLabel': 'string',
                                'srsName': 'string',
                                'axisLabels': [
                                    'string'
                                ],
                                'uomLabels': [
                                    'string'
                                ],
                                'coordinates': [
                                    [
                                        'string'
                                    ]
                                ]
                            },
                            'model': {
                                'type': 'TransformationBySensorModel',
                                'id': 'string',
                                'axisLabels': [
                                    'string'
                                ],
                                'uomLabels': [
                                    'string'
                                ],
                                'sensorModelRef': 'string',
                                'sensorInstanceRef': 'string'
                            },
                            'gridLimits': {
                                'type': 'GridLimits',
                                'indexAxis': {
                                    'type': 'IndexAxis',
                                    'id': 'string',
                                    'axisLabel': 'string',
                                    'lowerBound': 0,
                                    'upperBound': 0
                                },
                                'srsName': 'string',
                                'axisLabels': [
                                    'string'
                                ]
                            }
                        }
                    },
                    'rangeSet': {
                        'type': 'RangeSet',
                        'dataBlock': {
                            'type': 'VDataBlock',
                            'values': [
                                'string'
                            ]
                        }
                    },
                    'rangeType': {
                        'type': 'DataRecord',
                        'field': [
                            {
                                'type': 'Quantity',
                                'id': 'string',
                                'name': 'string',
                                'definition': 'string',
                                'uom': {
                                    'type': 'UnitReference',
                                    'id': 'string',
                                    'code': 'string'
                                },
                                'constraint': {
                                    'type': 'AllowedValues',
                                    'id': 'string',
                                    'interval': [
                                        'string'
                                    ]
                                }
                            }
                        ],
                        'interpolationRestriction': {
                            'type': 'InterpolationRestriction',
                            'id': 'string',
                            'allowedInterpolation': [
                                'string'
                            ]
                        }
                    },
                    'metadata': {}
                }
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}/coverage/domainset")
class CatalogRootHandler(ApiHandler[CoveragesContext]):
    @api.operation(operation_id='coveragesDomainSet',
                   summary='OGC API - Coverages - domain set')
    async def get(self, collectionId: str):
        domain_set = get_coverage_domainset(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(domain_set)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route("/catalog/collections/{collectionId}/coverage/rangetype")
class CatalogRootHandler(ApiHandler[CoveragesContext]):
    # noinspection PyPep8Naming
    @api.operation(operation_id='coveragesDomainSet',
                   summary='OGC API - Coverages - range type')
    async def get(self, collectionId: str):
        range_type = get_coverage_rangetype(self.ctx.datasets_ctx, collectionId)
        return self.response.finish(range_type)
