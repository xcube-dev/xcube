from unittest import TestCase

from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_false
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_in
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_not_none
from xcube.util.assertions import assert_subclass
from xcube.util.assertions import assert_true


class AssertTest(TestCase):
    def test_assert_not_none(self):
        assert_not_none(10, 'x')
        with self.assertRaises(ValueError) as e:
            assert_not_none(None, 'x')
        self.assertEqual(('x must not be None',), e.exception.args)

    def test_assert_given(self):
        assert_given('data.txt', 'x')
        with self.assertRaises(ValueError) as e:
            assert_given('', 'x')
        self.assertEqual(('x must be given',), e.exception.args)

    def test_assert_instance(self):
        assert_instance(10, int, 'x')
        assert_instance('data.txt', (int, str), 'x')
        with self.assertRaises(TypeError) as e:
            assert_instance(0.5, (int, str), 'x')
        self.assertEqual(("x must be an instance of "
                          "(<class 'int'>, <class 'str'>), "
                          "was <class 'float'>",),
                         e.exception.args)

    def test_assert_subclass(self):
        class A:
            pass

        class B(A):
            pass

        assert_subclass(A, A, 'x')
        assert_subclass(B, A, 'x')
        assert_subclass(B, (A, B), 'x')
        with self.assertRaises(TypeError):
            assert_subclass(4, A, 'x')
        with self.assertRaises(TypeError) as e:
            assert_subclass(A, B, 'x')
        self.assertEqual(
            ("x must be a subclass of"
             " <class 'test.util.test_assert.AssertTest"
             ".test_assert_subclass.<locals>.B'>,"
             " was"
             " <class 'test.util.test_assert.AssertTest"
             ".test_assert_subclass.<locals>.A'>",),
            e.exception.args)

    def test_assert_in(self):
        assert_in(2, (1, 2, 3), 'x')
        with self.assertRaises(ValueError) as e:
            assert_in(4, (1, 2, 3), 'x')
        self.assertEqual(('x must be one of (1, 2, 3)',), e.exception.args)

    def test_assert_true(self):
        assert_true(True, 'Should be true')
        with self.assertRaises(ValueError) as e:
            assert_true(False, 'Should be true')
        self.assertEqual(('Should be true',), e.exception.args)

    def test_assert_false(self):
        assert_false(False, 'Should be false')
        with self.assertRaises(ValueError) as e:
            assert_false(True, 'Should be false')
        self.assertEqual(('Should be false',), e.exception.args)

    def test_assert_condition(self):
        assert_condition(True, 'Should be true')
        with self.assertRaises(ValueError) as e:
            assert_condition(False, 'Should be true')
        self.assertEqual(('Should be true',), e.exception.args)
