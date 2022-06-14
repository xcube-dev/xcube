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

from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonStringSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonBooleanSchema

"""
Authentication:
  Domain: xcube-dev.eu.auth0.com
  Audience: https://xcube-dev/api/
  IsRequired: 
  
AccessControl:
  RequiredScopes:
    # Clients must be granted permission "read:dataset:l2c-cyanoalert-olci-balt" in auth0
    # to be able to see this dataset
    - "read:dataset:{Dataset}"
    # Clients must be granted permission "read:variable:chl_c2rcc" in auth0
    # to be able to see variable "chl_c2rcc"
    - "read:variable:{Variable}"  
"""

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        Authentication=JsonObjectSchema(
            properties=dict(
                Domain=JsonStringSchema(min_length=1, format='uri'),
                Audience=JsonStringSchema(min_length=1, format='uri'),
                Algorithms=JsonArraySchema(
                    items=JsonStringSchema(const=["RS256"]),
                ),
                IsRequired=JsonBooleanSchema(),
            ),
            required=["Domain", "Audience"],
            additional_properties=False
        ),
        AccessControl=JsonObjectSchema(
            properties=dict(
                RequiredScopes=JsonArraySchema(
                    items=JsonStringSchema(min_length=1)
                ),
            ),
            additional_properties=False
        ),
    ),
    additional_properties=False
)
