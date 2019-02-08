# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import re
from typing import Generator

import xarray as xr

Token = collections.namedtuple('Token', ['kind', 'value'])

_TOKEN_REGEX = None
_KW_MAPPINGS = {'NOT': 'not', 'AND': 'and', 'OR': 'or', 'true': 'True', 'false': 'False', 'NaN': 'NaN'}
_OP_MAPPINGS = {'?': 'if', ':': 'else', '!': 'not', '&&': 'and', '||': 'or'}


def translate_snap_expr_attributes(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dataset.copy()
    for var_name in dataset.variables:
        var = dataset[var_name]
        if 'expression' in var.attrs:
            var.attrs['expression'] = translate_snap_expr(var.attrs['expression'])
        if 'valid_pixel_expression' in var.attrs:
            var.attrs['valid_pixel_expression'] = translate_snap_expr(var.attrs['valid_pixel_expression'])
    return dataset


def _get_token_regex():
    global _TOKEN_REGEX
    if _TOKEN_REGEX is None:
        token_specification = [
            ('NUM', r'\d+(\.\d*)?'),  # Integer or decimal number
            ('ID', r'[_A-Za-z][_A-Za-z0-9]*'),  # Identifiers
            ('OP', r'\&\&|\|\||\*\*|\!\=|\=\=|\>=|\<=|\<|\>|\+|\-|\*|\!|\%|\^|\.|\?|\:|\||\&'),  # Operators
            ('PAR', r'[\(\)]'),  # Parentheses
            ('WHITE', r'[ \t\n\r]+'),  # Skip over spaces and tabs, new lines
            ('ERR', r'.'),  # Any other character
        ]
        _TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        _TOKEN_REGEX = re.compile(_TOKEN_REGEX)
    return _TOKEN_REGEX


def translate_snap_expr(snap_expr: str) -> str:
    """
    Translate a SNAP band math expression to a syntactically correct Python expression.

    :param snap_expr: SNAP band math expression
    :return: Python expression
    """
    py_expr = ''
    last_kind = None
    for token in tokenize_expr(snap_expr):
        kind = token.kind
        value = token.value
        if kind == 'ID' or kind == 'KW' or kind == 'NUM':
            value = _KW_MAPPINGS.get(value, value)
            if last_kind == 'ID' or last_kind == 'KW' or last_kind == 'NUM':
                py_expr += ' '
        elif kind == 'OP':
            if last_kind == 'OP':
                py_expr += ' '
        elif kind == 'PAR':
            pass
        py_expr += value
        last_kind = token.kind
    return py_expr


def tokenize_expr(expr: str) -> Generator[Token, None, None]:
    token_regex = _get_token_regex()
    for match in re.finditer(token_regex, expr):
        kind = match.lastgroup
        value = match.group(kind)
        if kind == 'WHITE':
            pass
        elif kind == 'ERR':
            raise RuntimeError(f'{value!r} unexpected in expression {expr!r}')
        else:
            if kind == 'ID' and value in _KW_MAPPINGS:
                kind = 'KW'
            elif kind == 'OP':
                # In order to deal with SNAP conditional expr, we tokenize '?' to 'if' and ':' to 'else'
                # so we can later build a syntactically valid Python expression. However we will need a special
                # treatment because the semantic isn't right that way:
                #
                #    a ? b : c  --> a if b else c --> b if a else c
                #
                kw = _OP_MAPPINGS.get(value)
                if kw is not None:
                    kind = 'KW'
                    value = kw
            yield Token(kind, value)
