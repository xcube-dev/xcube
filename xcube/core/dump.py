# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr


def dump_dataset(dataset: xr.Dataset, var_names=None, show_var_encoding=False) -> str:
    """Dumps a dataset or variables into a text string.

    Args:
        dataset: input dataset
        var_names: names of variables to be dumped
        show_var_encoding: also dump variable encodings?

    Returns:
        the dataset dump
    """
    lines = []
    if not var_names:
        lines.append(str(dataset))
        if show_var_encoding:
            for var_name, var in dataset.coords.items():
                if var.encoding:
                    lines.append(
                        dump_var_encoding(
                            var,
                            header=f"Encoding for coordinate variable {var_name!r}:",
                        )
                    )
            for var_name, var in dataset.data_vars.items():
                if var.encoding:
                    lines.append(
                        dump_var_encoding(
                            var, header=f"Encoding for data variable {var_name!r}:"
                        )
                    )
    else:
        for var_name in var_names:
            var = dataset[var_name]
            lines.append(str(var))
            if show_var_encoding and var.encoding:
                lines.append(dump_var_encoding(var))
    return "\n".join(lines)


def dump_var_encoding(var: xr.DataArray, header="Encoding:", indent=4) -> str:
    """Dump the encoding information of a variable into a text string.

    Args:
        var: Dataset variable.
        header: Title/header string.
        indent: Indention in spaces.

    Returns:
        the variable dump
    """
    lines = [header]
    max_len = 0
    for k in var.encoding:
        max_len = max(max_len, len(k))
    indent_spaces = indent * " "
    for k, v in var.encoding.items():
        tab_spaces = (2 + max_len - len(k)) * " "
        lines.append(f"{indent_spaces}{k}:{tab_spaces}{v!r}")
    return "\n".join(lines)
