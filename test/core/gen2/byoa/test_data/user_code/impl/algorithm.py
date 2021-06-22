from typing import Any, Dict

import numpy as np
import xarray as xr


def compute_chunk(input_var_1: np.ndarray,
                  input_var_2: np.ndarray,
                  input_params: Dict[str, Any]) -> xr.Dataset:
    factor_1: float = input_params['factor_1']
    factor_2: float = input_params['factor_2']
    return factor_1 * input_var_1 + factor_2 * input_var_2
