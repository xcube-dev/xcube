import unittest

from xcube.util.ipython import register_json_formatter


class IPythonTest(unittest.TestCase):

    def test_it(self):
        class _FormatterTest1:
            def to_dict(self):
                return dict()

        # Should work
        register_json_formatter(_FormatterTest1)

        class _FormatterTest2:
            def to_dictomat(self):
                return dict()

        # Should not work
        with self.assertRaises(ValueError) as cm:
            register_json_formatter(_FormatterTest2)
        self.assertTrue(f'{cm.exception}'.endswith("FormatterTest2'> must define a to_dict() method"),
                        msg=f'{cm.exception}')
