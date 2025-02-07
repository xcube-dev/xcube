# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr
from impl.algorithm import compute_chunk

from xcube.core.compute import compute_dataset


def process_dataset(
    dataset: xr.Dataset,
    output_var_name: str,
    input_var_name_1: str,
    input_var_name_2: str,
    factor_1: float = 1.0,
    factor_2: float = 2.0,
) -> xr.Dataset:
    return compute_dataset(
        compute_chunk,
        dataset,
        input_var_names=[input_var_name_1, input_var_name_2],
        input_params=dict(factor_1=factor_1, factor_2=factor_2),
        output_var_name=output_var_name,
    )
