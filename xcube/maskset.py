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
        flag_masks = flag_var.attrs.get('flag_masks')
        flag_values = flag_var.attrs.get('flag_values')
        if flag_masks is None and flag_values is None:
            raise ValueError("flag_var must either have one of the attributes 'flag_meanings' or 'flag_values' or both")
        _check_flag_var_attribute_value(flag_masks, 'flag_masks')
        _check_flag_var_attribute_value(flag_values, 'flag_values')
        flag_meanings = flag_var.attrs.get('flag_meanings')
        if not isinstance(flag_meanings, str):
            raise TypeError("attribute 'flag_meanings' of flag_var must be a string")
        flag_names = flag_meanings.split(' ')
        if flag_masks is not None and len(flag_names) != len(flag_masks):
            raise ValueError("attributes 'flag_meanings' and 'flag_masks' are not corresponding")
        if flag_values is not None and len(flag_names) != len(flag_values):
            raise ValueError("attributes 'flag_meanings' and 'flag_values' are not corresponding")
        if flag_masks is None:
            flag_masks = [None] * len(flag_names)
        if flag_values is None:
            flag_values = [None] * len(flag_names)
        self._flag_var = flag_var
        self._flag_names = flag_names
        self._flags = dict(zip(flag_names, list(zip(flag_masks, flag_values))))
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
        return "%s(%s)" % (self._flag_var.name, ', '.join(["%s=%s" % (n, v) for n, v in self._flags.items()]))

    def __getattr__(self, name: str) -> Any:
        if name not in self._flags:
            raise AttributeError(name)
        return self.get_mask(name)

    def __getitem__(self, item):
        try:
            name = self._flag_names[item]
            if name not in self._flags:
                raise IndexError(item)
        except TypeError:
            name = item
            if name not in self._flags:
                raise KeyError(item)
        return self.get_mask(name)

    def get_mask(self, flag_name: str):
        if flag_name not in self._flags:
            raise ValueError('invalid flag name "%s"' % flag_name)

        if flag_name in self._masks:
            return self._masks[flag_name]

        flag_var = self._flag_var
        flag_mask, flag_value = self._flags[flag_name]

        mask_var = xr.DataArray(np.ones(flag_var.shape, dtype=np.uint8), dims=flag_var.dims, name=flag_name)
        if flag_mask is not None:
            if flag_value is not None:
                mask_var = mask_var.where((flag_var & flag_mask) == flag_value, 0)
            else:
                mask_var = mask_var.where((flag_var & flag_mask) != 0, 0)
        else:
            mask_var = mask_var.where(flag_var == flag_value, 0)

        self._masks[flag_name] = mask_var
        return mask_var


def _check_flag_var_attribute_value(attr_value, attr_name):
    if attr_value is None:
        return
    if not (hasattr(attr_value, 'dtype')
            and hasattr(attr_value, 'shape')
            and np.issubdtype(attr_value.dtype, np.integer)):
        raise TypeError(f'attribute {attr_name!r} of flag_var must be an integer array')
