# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

from typing import Dict, Any

import numpy as np
import xarray as xr


# TODO: this would be useful to have in xarray:
#       >>> ds = xr.open_dataset("my/path/to/cf/netcdf.nc", decode_flags=True)
#       >>> ds.flag_mask_sets['quality_flags']

class MaskSet:
    """
    A set of mask variables derived from a variable *flag_var* with
    CF attributes "flag_masks" and "flag_meanings".

    Each mask is represented by an `xarray.DataArray` and
    has the name of the flag, is of type `numpy.unit8`, and has the dimensions
    of the given *flag_var*.

    :param flag_var: an `xarray.DataArray` that defines flag values.
           The CF attributes "flag_masks" and "flag_meanings" are expected to
           exists and be valid.
    """

    def __init__(self, flag_var: xr.DataArray):
        flag_masks = flag_var.attrs['flag_masks']
        if flag_masks is None \
                or not hasattr(flag_masks, 'dtype') \
                or not hasattr(flag_masks, 'shape') \
                or not np.issubdtype(flag_masks.dtype, np.integer):
            raise TypeError("'flag_meanings' attribute of flag_var must be an integer array")
        flag_meanings = flag_var.attrs['flag_meanings']
        if not isinstance(flag_meanings, str):
            raise TypeError("'flag_meanings' attribute of flag_var must be a string")
        flag_names = flag_meanings.split(' ')
        if len(flag_names) != len(flag_masks):
            raise ValueError("'flag_names' and 'flag_meanings' are not corresponding")
        self._flag_var = flag_var
        self._flag_names = flag_names
        self._flag_values = dict(zip(flag_names, flag_masks))
        self._masks = {}

    @classmethod
    def get_mask_sets(cls, dataset: xr.Dataset) -> Dict[str, 'MaskSet']:
        """
        For each "flag" variable in given *dataset*, turn it into a ``MaskSet``, store it in a dictionary.

        :param dataset: The dataset
        :return: A mapping of flag names to ``MaskSet``. Will be empty if there are no flag variables in *dataset*.
        """
        masks = {}
        for var_name in dataset.variables:
            var = dataset[var_name]
            if 'flag_masks' in var.attrs and 'flag_meanings' in var.attrs:
                masks[var_name] = MaskSet(var)
        return masks

    def __str__(self):
        return "%s(%s)" % (self._flag_var.name, ', '.join(["%s=%s" % (n, v) for n, v in self._flag_values.items()]))

    def __getattr__(self, name: str) -> Any:
        if name not in self._flag_values:
            raise AttributeError(name)
        return self.get_mask(name)

    def __getitem__(self, item):
        try:
            name = self._flag_names[item]
            if name not in self._flag_values:
                raise IndexError(item)
        except TypeError:
            name = item
            if name not in self._flag_values:
                raise KeyError(item)
        return self.get_mask(name)

    def get_mask(self, flag_name: str):
        if flag_name not in self._flag_values:
            raise ValueError('invalid flag name "%s"' % flag_name)

        if flag_name in self._masks:
            return self._masks[flag_name]

        flag_var = self._flag_var
        flag_value = self._flag_values[flag_name]

        mask_var = xr.DataArray(np.ones(flag_var.shape, dtype=np.uint8), dims=flag_var.dims, name=flag_name)
        mask_var = mask_var.where((flag_var & flag_value) != 0, 0)
        self._masks[flag_name] = mask_var
        return mask_var
