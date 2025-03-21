# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Dict, Set

import xarray as xr

ANCILLARY_VAR_NAME_PREFIXES_TO_STANDARD_NAME_MODIFIERS = [
    ("std", "standard_error"),
    ("count", "number_of_observations"),
]


def find_ancillary_var_names(
    dataset: xr.Dataset,
    var_name: str,
    same_shape: bool = False,
    same_dims: bool = False,
) -> dict[str, set[str]]:
    variable = dataset.data_vars.get(var_name)

    results = {}

    # Check for CF compatibility according to
    # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#ancillary-data
    #
    if variable is not None and "ancillary_variables" in variable.attrs:
        ancillary_var_names = variable.attrs["ancillary_variables"].split(" ")
        for ancillary_var_name in ancillary_var_names:
            if ancillary_var_name in dataset.data_vars:
                ancillary_var = dataset.data_vars[ancillary_var_name]
                if (not same_shape or variable.shape == ancillary_var.shape) and (
                    not same_dims or variable.dims == variable.dims
                ):
                    standard_name_modifier = (
                        _get_standard_name_modifier(variable, ancillary_var) or ""
                    )
                    if standard_name_modifier not in results:
                        results[standard_name_modifier] = set()
                    results[standard_name_modifier].add(ancillary_var_name)

    if variable is not None and not results:
        #
        # Check for less strict CF compatibility (missing attribute 'ancillary_variables')
        #
        if "standard_name" in variable.attrs:
            for ancillary_var_name, ancillary_var in dataset.data_vars.items():
                if ancillary_var is variable:
                    continue
                standard_name_modifier = _get_standard_name_modifier(
                    variable, ancillary_var
                )
                if standard_name_modifier is not None:
                    if standard_name_modifier not in results:
                        results[standard_name_modifier] = set()
                    results[standard_name_modifier].add(ancillary_var_name)

    if not results:
        #
        # Search for variables with xcube-specific prefixes that indicate uncertainty:
        #
        for (
            prefix,
            standard_name_modifier,
        ) in ANCILLARY_VAR_NAME_PREFIXES_TO_STANDARD_NAME_MODIFIERS:
            ancillary_var_name = f"{var_name}_{prefix}"
            if ancillary_var_name in dataset.data_vars:
                if standard_name_modifier not in results:
                    results[standard_name_modifier] = set()
                results[standard_name_modifier].add(ancillary_var_name)

    return results


def _get_standard_name_modifier(variable, ancillary_var):
    """See CF Conventions v 1.7, Appendix C: Standard Name Modifiers"""
    if "standard_name" in ancillary_var.attrs:
        ancillary_var_std_name = ancillary_var.attrs["standard_name"]
        ancillary_var_std_name_parts = ancillary_var_std_name.split(" ")
        if len(ancillary_var_std_name_parts) == 2 and ancillary_var_std_name_parts[
            0
        ] == variable.attrs.get("standard_name"):
            return ancillary_var_std_name_parts[-1]
    return None
