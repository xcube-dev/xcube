# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
from xcube.util.jsonencoder import to_json_value

PY_INPUT = {
    'py_bool': True,
    'py_int': 11,
    'py_float': 12.3,
    'py_str': 'Hallo',
    'py_null': None,
    'py_list': [1, 2, 3],
    'py_dict': {'x': 'A', 'y': 'B'},
}

NP_INPUT = {
    'np_bool': True,
    'np_int8': np.int8(1),
    'np_uint8': np.uint8(2),
    'np_int16': np.int16(3),
    'np_uint16': np.uint8(4),
    'np_int32': np.int32(5),
    'np_uint32': np.uint32(6),
    'np_int64': np.int64(7),
    'np_uint64': np.uint64(8),
    'np_float32': np.float32(9.1),
    'np_float64': np.float64(9.2),
    'np_uint8_array': np.array([1, 2, 3], dtype=np.uint8),
    'np_float64_array': np.array([0.1, 0.2, 0.3], dtype=np.float64),
}

INPUT = {
    **PY_INPUT,
    **NP_INPUT,
}

EXPECTED_PY_OUTPUT = {**PY_INPUT}

EXPECTED_NP_OUTPUT = {
    'np_bool': True,
    'np_int8': int(np.int8(1)),
    'np_uint8': int(np.uint8(2)),
    'np_int16': int(np.int16(3)),
    'np_uint16': int(np.uint8(4)),
    'np_int32': int(np.int32(5)),
    'np_uint32': int(np.uint32(6)),
    'np_int64': int(np.int64(7)),
    'np_uint64': int(np.uint64(8)),
    'np_float32': float(np.float32(9.1)),
    'np_float64': float(np.float64(9.2)),
    'np_uint8_array': [1, 2, 3],
    'np_float64_array': [0.1, 0.2, 0.3],
}

EXPECTED_OUTPUT = {
    **EXPECTED_PY_OUTPUT,
    **EXPECTED_NP_OUTPUT
}


class NumpyJSONEncoderTest(unittest.TestCase):

    # noinspection PyMethodMayBeStatic
    def test_fail_without_encoder(self):
        with pytest.raises(TypeError):
            json.dumps(INPUT, indent=2)

    def test_encoder_encodes_all(self):
        text = json.dumps(INPUT, indent=2, cls=NumpyJSONEncoder)
        data = json.loads(text)
        self.assertEqual(EXPECTED_OUTPUT, data)


class ToJsonValueTest(unittest.TestCase):
    def test_items(self):
        for k in PY_INPUT.keys():
            self.assertIs(PY_INPUT[k],
                          to_json_value(PY_INPUT[k]))
        for k in INPUT.keys():
            self.assertEqual(EXPECTED_OUTPUT[k],
                             to_json_value(INPUT[k]))

    def test_dict(self):
        self.assertIs(PY_INPUT,
                      to_json_value(PY_INPUT))

        self.assertIsNot(INPUT,
                         to_json_value(INPUT))
        self.assertEqual(EXPECTED_OUTPUT,
                         to_json_value(INPUT))

        input_dict = dict(a=PY_INPUT, b=PY_INPUT)
        self.assertIs(input_dict,
                      to_json_value(input_dict))

        input_dict = dict(a=INPUT, b=INPUT)
        self.assertIsNot(input_dict,
                         to_json_value(input_dict))
        self.assertEqual(dict(a=EXPECTED_OUTPUT, b=EXPECTED_OUTPUT),
                         to_json_value(input_dict))

        with pytest.raises(TypeError,
                           match='Property names of JSON objects'
                                 ' must be strings, but got bool'):
            to_json_value({True: 13})

    def test_list(self):
        input_list = [PY_INPUT, PY_INPUT]
        self.assertIs(input_list,
                      to_json_value(input_list))

        input_list = [INPUT, INPUT]
        self.assertIsNot(input_list,
                         to_json_value(input_list))
        self.assertEqual([EXPECTED_OUTPUT,
                          EXPECTED_OUTPUT],
                         to_json_value(input_list))

        input_tuple = (INPUT, INPUT)
        self.assertEqual([EXPECTED_OUTPUT,
                          EXPECTED_OUTPUT],
                         to_json_value(input_tuple))

    def test_numpy_arrays(self):
        import numpy

        array = numpy.array([1, 2, 3])
        self.assertIsNot(array, to_json_value(array))
        self.assertEqual([1, 2, 3], to_json_value(array))

        array = numpy.array([[1, 2], [3, 4]], dtype=np.uint8)
        self.assertIsNot(array, to_json_value(array))
        self.assertEqual([[1, 2], [3, 4]], to_json_value(array))

        array = numpy.array([], dtype=np.float32)
        self.assertIsNot(array, to_json_value(array))
        self.assertEqual([], to_json_value(array))

        array = numpy.array(["2020-01-02 10:00:05",
                             "2020-01-03 14:10:36"], dtype=np.datetime64)
        self.assertIsNot(array, to_json_value(array))
        self.assertEqual(['2020-01-02T10:00:05Z',
                          '2020-01-03T14:10:36Z'], to_json_value(array))

    # noinspection PyMethodMayBeStatic
    def test_fails_correctly(self):
        with pytest.raises(TypeError,
                           match='Object of type object'
                                 ' is not JSON serializable'):
            to_json_value(object())
