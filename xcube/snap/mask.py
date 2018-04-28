import warnings
from typing import Tuple, Dict, Generator

import numpy as np
import xarray as xr

from xcube.maskset import MaskSet
from xcube.snap.transexpr import transpile_expr, tokenize_expr


# TODO: add option to save computed mask in output dataset

def mask_dataset(dataset: xr.Dataset,
                 expr_pattern: str = None,
                 in_place: bool = False,
                 errors: str = None) -> Tuple[xr.Dataset, Dict[str, MaskSet]]:
    """
    BEAM/SNAP specific dataset masking.
    Evaluate "valid_pixel_expression" and fill-in NaN where invalid.

    :param dataset: xarray dataset read from netCDF produced with BEAM or SNAP.
    :param expr_pattern: if given, valid-pixel-expression will be processed through the pattern which should contain "{expr}".
    :param in_place: change dataset in-plase or make shallow copy
    :param errors: "raise", "warn", or "ignore"
    :return: new dataset with masked variables
    """
    errors = errors or 'raise'

    # Setup namespace used to evaluate transpiled SNAP expressions
    namespace = dict(NaN=np.nan, np=np, xr=xr)
    # Add all variables to namespace that are not expression variables such as SNAP masks and virtual bands
    for var_name in dataset.variables:
        var = dataset[var_name]
        if not _is_snap_expression_var(var):
            # TODO: only add spatial variables, exclude tie point grids
            namespace[var_name] = var

    # Add to namespace all SNAP flag bands as mask sets, where each mask set has a mask for each flag
    mask_sets = MaskSet.get_mask_sets(dataset)
    # for mask_name, mask_set in mask_sets.items():
    #     print(mask_name, mask_set)
    namespace.update(mask_sets)

    # Evaluate all variables that represent SNAP masks and virtual bands and also add to namespace
    for var_name in dataset.variables:
        var = dataset[var_name]
        if _is_snap_expression_var(var):
            snap_expression = var.attrs['expression']
            computed_var = _compute_snap_expression(var_name, snap_expression, namespace, errors)
            if computed_var is not None:
                namespace[var_name] = computed_var

    # Now that the namespace is ready, evaluate and apply valid-pixel-expression masks
    masked_dataset = dataset if in_place else dataset.copy()
    for var_name in dataset.variables:
        var = dataset[var_name]
        if 'valid_pixel_expression' in var.attrs:
            snap_expression = var.attrs['valid_pixel_expression']
            if expr_pattern:
                snap_expression = expr_pattern.format(expr=snap_expression)
            valid_mask = _compute_snap_expression(var_name, snap_expression, namespace, errors)
            if valid_mask is not None:
                masked_var = var.where(valid_mask)
                masked_dataset[var_name] = masked_var

    return masked_dataset, mask_sets


def _compute_snap_expression(var_name, snap_expression, namespace, errors):
    numpy_expression = _snap_expr_to_numpy_expr(snap_expression, errors)
    # print('variable "%s" uses expression %r transpiled to %r' % (var_name, snap_expression, numpy_expression))
    try:
        computed_var = eval(numpy_expression, namespace, None)
    except Exception as e:
        computed_var = None
        msg = 'error evaluating variable "%s" with expression "%s" transpiled from SNAP expression "%s": ' \
              '%s' % (var_name, numpy_expression, snap_expression, e)
        if errors == 'raise':
            raise RuntimeError(msg) from e
        elif errors == 'warn':
            warnings.warn(msg)
    return computed_var


def _is_snap_expression_var(var: xr.DataArray) -> bool:
    return 'expression' in var.attrs and var.shape == ()


def _snap_expr_to_numpy_expr(snap_expr: str, errors: str) -> str:
    py_expr = _translate_expr(snap_expr)
    return transpile_expr(py_expr, warn=errors == 'raise' or errors == 'warn')


def _translate_expr(ba_expr: str) -> str:
    py_expr = ''
    translations = {'NOT': 'not', 'AND': 'and', 'OR': 'or', 'true': 'True', 'false': 'False'}
    last_kind = None
    for token in tokenize_expr(ba_expr):
        kind = token.kind
        value = token.value
        if kind == 'ID' or kind == 'KW' or kind == 'NUM':
            value = translations.get(value, value)
            if last_kind == 'ID' or last_kind == 'KW' or last_kind == 'NUM':
                py_expr += ' '
        elif kind == 'OP':
            if last_kind == 'OP':
                py_expr += ' '
            pass
        elif kind == 'PAR':
            pass
        py_expr += value
        last_kind = token.kind
    return py_expr
