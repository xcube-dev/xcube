# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
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

import json
import unittest

import numpy as np
import pytest

from xcube.util.jsonencoder import NumpyJSONEncoder


class NumpyJSONEncoderTest(unittest.TestCase):
    TEST_DATA = {
        "np_bool": np.bool(True),
        "np_int8": np.int8(1),
        "np_uint8": np.uint8(2),
        "np_int16": np.int16(3),
        "np_uint16": np.uint8(4),
        "np_int32": np.int32(5),
        "np_uint32": np.uint32(6),
        "np_int64": np.int64(7),
        "np_uint64": np.uint64(8),
        "np_float32": np.float32(9.1),
        "np_float64": np.float64(9.2),
        "py_bool": True,
        "py_int": 11,
        "py_float": 12.3,
        "py_str": "Hallo",
        "py_null": None,
    }

    def test_fail_without_encoder(self):
        with pytest.raises(TypeError):
            json.dumps(
                self.TEST_DATA,
                indent=2,
            )

    def test_encoder_encodes_all(self):
        text = json.dumps(
            self.TEST_DATA,
            indent=2,
            cls=NumpyJSONEncoder
        )
        data = json.loads(text)
        self.assertEqual(
            {
                "np_bool": bool(np.bool(True)),
                "np_int8": int(np.int8(1)),
                "np_uint8": int(np.uint8(2)),
                "np_int16": int(np.int16(3)),
                "np_uint16": int(np.uint8(4)),
                "np_int32": int(np.int32(5)),
                "np_uint32": int(np.uint32(6)),
                "np_int64": int(np.int64(7)),
                "np_uint64": int(np.uint64(8)),
                "np_float32": float(np.float32(9.1)),
                "np_float64": float(np.float64(9.2)),
                "py_bool": True,
                "py_int": 11,
                "py_float": 12.3,
                "py_str": "Hallo",
                "py_null": None,
            },
            data
        )
