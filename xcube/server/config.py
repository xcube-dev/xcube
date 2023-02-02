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

from typing import Mapping, Any, Optional

from xcube.constants import DEFAULT_SERVER_ADDRESS
from xcube.constants import DEFAULT_SERVER_PORT
from xcube.util.jsonschema import JsonArraySchema
from xcube.util.jsonschema import JsonBooleanSchema
from xcube.util.jsonschema import JsonIntegerSchema
from xcube.util.jsonschema import JsonObjectSchema
from xcube.util.jsonschema import JsonStringSchema

BASE_SERVER_CONFIG_SCHEMA = JsonObjectSchema(
    properties=dict(
        port=JsonIntegerSchema(
            title='Server port.',
            default=DEFAULT_SERVER_PORT
        ),
        address=JsonStringSchema(
            title='Server address.',
            default=DEFAULT_SERVER_ADDRESS
        ),
        base_dir=JsonStringSchema(
            title='Base directory used to resolve relative local paths.',
        ),
        url_prefix=JsonStringSchema(
            title='Prefix to be prepended to all URL route paths.',
        ),
        reverse_url_prefix=JsonStringSchema(
            title='Prefix to be prepended to reverse URL paths'
                  ' returned by server responses.',
        ),
        trace_perf=JsonBooleanSchema(
            title='Output performance measures',
        ),
        static_routes=JsonArraySchema(
            title='Static content routes',
            items=JsonObjectSchema(
                properties=dict(
                    path=JsonStringSchema(
                        title='The URL path',
                        min_length=1
                    ),
                    dir_path=JsonStringSchema(
                        title='A local directory path',
                        min_length=1
                    ),
                    default_filename=JsonStringSchema(
                        title='Optional default filename',
                        examples=['index.html'],
                        min_length=1
                    ),
                    openapi_metadata=JsonObjectSchema(
                        title='Optional OpenAPI operation metadata',
                        additional_properties=True
                    ),
                ),
                required=['path', 'dir_path'],
                additional_properties=False
            )
        ),
        api_spec=JsonObjectSchema(
            title='API specification',
            description='selected = (includes | ALL) - (excludes | NONE)',
            properties=dict(
                includes=JsonArraySchema(
                    JsonStringSchema(min_length=1)
                ),
                excludes=JsonArraySchema(
                    JsonStringSchema(min_length=1)
                ),
            ),
            additional_properties=False
        )
    ),
    # We allow for other configuration settings contributed
    # by APIs. If these APIs are currently not in use,
    # validation would fail if additional_properties=False.
    additional_properties=True,
)


def get_url_prefix(config: Mapping[str, Any]) -> str:
    """
    Get the sanitized URL prefix so, if given, it starts with
    a leading slash and ends without one.

    :param config: Server configuration.
    :return: Sanitized URL prefix, may be an empty string.
    """
    return _sanitize_url_prefix(config.get('url_prefix'))


def get_reverse_url_prefix(config: Mapping[str, Any]) -> str:
    """
    Get the sanitized reverse URL prefix so, if given, it starts with
    a leading slash and ends without one.

    :param config: Server configuration.
    :return: Sanitized URL prefix, may be an empty string.
    """
    return _sanitize_url_prefix(
        config.get('reverse_url_prefix',
                   config.get('url_prefix'))
    )


def _sanitize_url_prefix(url_prefix: Optional[str]) -> str:
    """Get a sanitized URL prefix so, if given, it starts with
    a leading slash and ends without one.

    :param url_prefix: URL prefix path.
    :return: Sanitized URL prefix path, may be an empty string.
    """
    if not url_prefix:
        return ''
    while url_prefix.startswith('//'):
        url_prefix = url_prefix[1:]
    while url_prefix.endswith('/'):
        url_prefix = url_prefix[:-1]
    if url_prefix == '':
        return ''
    elif url_prefix.startswith('/'):
        return url_prefix
    else:
        return '/' + url_prefix
