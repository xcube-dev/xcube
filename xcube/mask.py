import warnings
from typing import Dict, Any

import numpy as np
import xarray as xr


def mask_dataset(dataset: xr.Dataset, in_place=False, errors=None):
    errors = errors or 'raise'
    mask_sets = LazyMaskSet.get_mask_sets(dataset)
    namespace = dict(NaN=np.nan)
    for var_name in dataset.variables:
        var = dataset[var_name]
        if not _is_ba_expr_var(var):
            namespace[var_name] = var
    namespace.update(mask_sets)
    for var_name in dataset.variables:
        var = dataset[var_name]
        if _is_ba_expr_var(var):
            expression = _ba_expr_to_py_expr(var.attrs['expression'])
            print(var_name, 'uses expression', expression)
            try:
                computed_var = eval(expression, namespace, None)
            except Exception as e:
                computed_var = None
                msg = 'error evaluating variable "%s" with expression "%s": ' \
                      '%s' % (var_name, expression, e)
                if errors == 'raise':
                    raise RuntimeError(msg) from e
                elif errors == 'warn':
                    warnings.warn(msg)

            if computed_var is not None:
                namespace[var_name] = computed_var

    masked_dataset = None if in_place else xr.Dataset()
    for var_name in dataset.variables:
        var = dataset[var_name]
        if 'valid_pixel_expression' in var.attrs:
            expression = _ba_expr_to_py_expr(var.attrs['valid_pixel_expression'])
            try:
                valid_mask = eval(expression, namespace, None)
            except Exception as e:
                valid_mask = None
                msg = 'error evaluating valid-mask for variable "%s" ' \
                      'with valid-pixel-expression "%s": %s' % (var_name, expression, e)
                if errors == 'raise':
                    raise RuntimeError(msg) from e
                elif errors == 'warn':
                    warnings.warn(msg)
            if valid_mask is not None:
                masked_var = var.where(valid_mask)
                if masked_dataset:
                    masked_dataset[var_name] = masked_var
                else:
                    dataset[var_name] = masked_var
        elif masked_dataset:
            masked_dataset[var_name] = var
    return dataset


def _is_ba_expr_var(var: xr.DataArray) -> bool:
    return 'expression' in var.attrs and var.shape == ()


def _ba_expr_to_py_expr(ba_expr: str) -> str:
    # TODO: replace this poor-man's translation from SNAP/BEAM band expressions
    #       to valid Python expressions by something more robust
    py_expr = ba_expr
    py_expr = py_expr.replace('!=', '___not_equal___')
    py_expr = py_expr.replace('!', ' not ')
    py_expr = py_expr.replace(' || ', ' or ')
    py_expr = py_expr.replace(' && ', ' and ')
    py_expr = py_expr.replace(' NOT ', ' not ')
    py_expr = py_expr.replace(' OR ', ' or ')
    py_expr = py_expr.replace(' AND ', ' and ')
    py_expr = py_expr.replace('___not_equal___', '!=')
    return py_expr


class LazyMaskSet:
    @classmethod
    def get_mask_sets(cls, dataset: xr.Dataset) -> Dict[str, 'LazyMaskSet']:
        masks = {}
        for var_name in dataset.variables:
            var = dataset[var_name]
            if 'flag_masks' in var.attrs and 'flag_meanings' in var.attrs:
                masks[var_name] = LazyMaskSet(var)
        return masks

    def __init__(self, flag_var: xr.DataArray):
        flag_masks = flag_var.attrs['flag_masks']
        flag_meanings = flag_var.attrs['flag_meanings']
        flag_names = flag_meanings.split(' ')
        self._flag_var = flag_var
        self._flag_names = flag_names
        self._flag_values = dict(zip(flag_names, flag_masks))
        self._masks = {}

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

        mask_var = xr.DataArray(np.zeros(flag_var.shape, dtype=np.uint8), dims=flag_var.dims, name=flag_name)
        mask_var = mask_var.where(flag_var & flag_value != 0, 1)
        self._masks[flag_name] = mask_var
        return mask_var
