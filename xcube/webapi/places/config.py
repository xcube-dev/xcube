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
from xcube.util.jsonschema import JsonObjectSchema
from xcube.webapi.common.schemas import IDENTIFIER_SCHEMA
from xcube.webapi.common.schemas import PATH_SCHEMA
from xcube.webapi.common.schemas import STRING_SCHEMA

PLACE_GROUP_JOIN_SCHEMA = JsonObjectSchema(
    properties=dict(
        Property=IDENTIFIER_SCHEMA,
        Path=PATH_SCHEMA,
    ),
    required=[
        'Property',
        'Path',
    ],
    additional_properties=False,
)

PLACE_GROUP_SCHEMA = JsonObjectSchema(
    properties=dict(
        Identifier=IDENTIFIER_SCHEMA,
        Title=STRING_SCHEMA,
        Path=PATH_SCHEMA,
        Query=STRING_SCHEMA,
        Join=PLACE_GROUP_JOIN_SCHEMA,
        CharacterEncoding=STRING_SCHEMA,
        PropertyMapping=JsonObjectSchema(
            additional_properties=PATH_SCHEMA
        ),
        PlaceGroupRef=IDENTIFIER_SCHEMA,
    ),
    required=[
        # Either we have
        #   'Identifier',
        #   'Path',
        # or we have
        #   'Identifier',
        #   'Query',
        # or we must specify
        #   'PlaceGroupRef',
    ],
    additional_properties=False,
)

CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        PlaceGroups=JsonArraySchema(items=PLACE_GROUP_SCHEMA)
    )
)
