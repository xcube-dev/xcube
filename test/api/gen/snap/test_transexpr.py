import unittest

from test.sampledata import create_highroc_dataset
from xcube.api.gen.snap.transexpr import tokenize_expr, Token, translate_snap_expr, translate_snap_expr_attributes


class TranslateExpressionAttributesTest(unittest.TestCase):
    def test_it(self):
        ds1 = create_highroc_dataset()
        ds2 = translate_snap_expr_attributes(ds1)
        self.assertIsNot(ds1, ds2)


class TranslateExprTest(unittest.TestCase):
    def test_translate_expr(self):
        self.assertEqual(translate_snap_expr('a'), 'a')
        self.assertEqual(translate_snap_expr('!a'), 'not a')
        self.assertEqual(translate_snap_expr('a && b'), 'a and b')
        self.assertEqual(translate_snap_expr('a || b'), 'a or b')
        self.assertEqual(translate_snap_expr('a & b'), 'a&b')
        self.assertEqual(translate_snap_expr('a | b'), 'a|b')


class TokenizeExprTest(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(list(tokenize_expr('a')),
                         [Token('ID', 'a')])
        self.assertEqual(list(tokenize_expr('true')),
                         [Token('KW', 'true')])
        self.assertEqual(list(tokenize_expr('234')),
                         [Token('NUM', '234')])
        self.assertEqual(list(tokenize_expr('234.2')),
                         [Token('NUM', '234.2')])
        self.assertEqual(list(tokenize_expr('a_2')),
                         [Token('ID', 'a_2')])

    def test_composites(self):
        self.assertEqual(list(tokenize_expr('a + b')),
                         [Token('ID', 'a'),
                          Token('OP', '+'),
                          Token('ID', 'b')])
        self.assertEqual(list(tokenize_expr('a > 0.5 AND(NOT b2 == true OR C._x != 3)')),
                         [Token('ID', 'a'),
                          Token('OP', '>'),
                          Token('NUM', '0.5'),
                          Token('KW', 'AND'),
                          Token('PAR', '('),
                          Token('KW', 'NOT'),
                          Token('ID', 'b2'),
                          Token('OP', '=='),
                          Token('KW', 'true'),
                          Token('KW', 'OR'),
                          Token('ID', 'C'),
                          Token('OP', '.'),
                          Token('ID', '_x'),
                          Token('OP', '!='),
                          Token('NUM', '3'),
                          Token('PAR', ')')])

    def test_conditional(self):
        self.assertEqual(list(tokenize_expr('a >= 0.0 ? a : NaN')),
                         [Token('ID', 'a'),
                          Token('OP', '>='),
                          Token('NUM', '0.0'),
                          Token('KW', 'if'),
                          Token('ID', 'a'),
                          Token('KW', 'else'),
                          Token('KW', 'NaN')])
