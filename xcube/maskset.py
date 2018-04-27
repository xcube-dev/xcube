from typing import Dict, Any

import numpy as np
import xarray as xr

# TODO: provide to xarray project

class MaskSet:
    """
    A set of mask variables. Each mask is represented by an xarray DataArray and
    has the name of the flag, is of type ``np.unit8``, and has the dimensions of the ``flag_var``.

    This is a general-purpose class that relies on CF conventions.

    :param flag_var: an xarray DataArray or Variable that defines flag values.
           The CF attributes "flag_masks" and "flag_meanings" are expected to
           exists and be valid.
    """

    def __init__(self, flag_var: xr.DataArray):
        flag_masks = flag_var.attrs['flag_masks']
        flag_meanings = flag_var.attrs['flag_meanings']
        flag_names = flag_meanings.split(' ')
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
