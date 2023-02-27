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


import unittest

import jsonschema.exceptions
import pytest

from xcube.webapi.viewer.config import CONFIG_SCHEMA


class ViewerConfigTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_validate_instance_ok(self):
        CONFIG_SCHEMA.validate_instance({
        })
        CONFIG_SCHEMA.validate_instance({
            "Viewer": {
            }
        })
        CONFIG_SCHEMA.validate_instance({
            "Viewer": {
                "Configuration": {
                },
            }
        })
        CONFIG_SCHEMA.validate_instance({
            "Viewer": {
                "Configuration": {
                    "Path": "s3://xcube-viewer-app/bc/dev/viewer/"
                },
            }
        })

    # noinspection PyMethodMayBeStatic
    def test_validate_instance_fails(self):
        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance({
                "Viewer": {
                    "Config": {
                    },
                }
            })

        with pytest.raises(jsonschema.exceptions.ValidationError):
            CONFIG_SCHEMA.validate_instance({
                "Viewer": {
                    "Configuration": {
                        "Path": "s3://xcube-viewer-app/bc/dev/viewer/",
                        "Title": "Test!"
                    },
                }
            })
