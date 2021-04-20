import unittest

import numpy as np
import xarray as xr

from xcube.core.ancvar import find_ancillary_var_names


class FindAncillaryVarNamesTest(unittest.TestCase):
    @classmethod
    def _new_array(self, dims=None, **attrs):
        dims = dims or [('time', 3), ('lat', 3), ('lon', 3)]
        s = 1
        for d in dims:
            s *= d[1]
        return xr.DataArray(np.linspace(0, 1, s).reshape([s for _, s in dims]), dims=([n for n, _ in dims]),
                            attrs=attrs)

    def test_strict_cf_convention(self):
        a = self._new_array(standard_name='a', ancillary_variables='a_stdev b c')
        a_stdev = self._new_array(standard_name='a standard_error')
        a_uncert = self._new_array(standard_name='a standard_error')
        a_count = self._new_array(standard_name='a number_of_observations')
        b = self._new_array(standard_name='b')
        b_stdev = self._new_array(standard_name='b standard_error')
        c = self._new_array(standard_name='c')

        ds = xr.Dataset(dict(a=a))
        self.assertEqual({}, find_ancillary_var_names(ds, 'a'))

        ds = xr.Dataset(dict(a=a, a_stdev=a_stdev, a_uncert=a_uncert, a_count=a_count, b=b, b_stdev=b_stdev, c=c))
        self.assertEqual({'': {'c', 'b'}, 'standard_error': {'a_stdev'}},
                         find_ancillary_var_names(ds, 'a'))

    def test_less_strict_cf_convention(self):
        a = self._new_array(standard_name='a')
        a_stdev = self._new_array(standard_name='a standard_error')
        a_uncert = self._new_array(standard_name='a standard_error')
        a_count = self._new_array(standard_name='a number_of_observations')
        b = self._new_array(standard_name='b')
        b_stdev = self._new_array(standard_name='b standard_error')

        ds = xr.Dataset(dict(a=a))
        self.assertEqual({}, find_ancillary_var_names(ds, 'a'))

        ds = xr.Dataset(dict(a=a, a_stdev=a_stdev, a_uncert=a_uncert, a_count=a_count, b=b, b_stdev=b_stdev))
        self.assertEqual({'number_of_observations': {'a_count'},
                          'standard_error': {'a_stdev', 'a_uncert'}},
                         find_ancillary_var_names(ds, 'a'))

    def test_xcube_prefix_convention(self):
        a = self._new_array()
        a_std = self._new_array()
        a_count = self._new_array()
        b = self._new_array()
        b_std = self._new_array()

        ds = xr.Dataset(dict(a=a))
        self.assertEqual({}, find_ancillary_var_names(ds, 'a'))

        ds = xr.Dataset(dict(a=a, a_std=a_std, a_count=a_count, b=b, b_std=b_std))
        self.assertEqual({'number_of_observations': {'a_count'},
                          'standard_error': {'a_std'}},
                         find_ancillary_var_names(ds, 'a'))
