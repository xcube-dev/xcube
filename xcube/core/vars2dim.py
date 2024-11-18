# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import xarray as xr

from xcube.core.verify import assert_cube


def vars_to_dim(
    cube: xr.Dataset,
    dim_name: str = "var",
    var_name="data",
    cube_asserted: bool = False,
):
    """Convert data variables into a dimension.

    Args:
        cube: The xcube dataset.
        dim_name: The name of the new dimension and coordinate variable.
            Defaults to 'var'.
        var_name: The name of the new, single data variable. Defaults to
            'data'.
        cube_asserted: If False, *cube* will be verified, otherwise it
            is expected to be a valid cube.

    Returns:
        A new xcube dataset with data variables turned into a new
        dimension.
    """

    if not cube_asserted:
        assert_cube(cube)

    if var_name == dim_name:
        raise ValueError("var_name must be different from dim_name")

    data_var_names = [data_var_name for data_var_name in cube.data_vars]
    if not data_var_names:
        raise ValueError("cube must not be empty")

    da = xr.concat([cube[data_var_name] for data_var_name in data_var_names], dim_name)
    new_coord_var = xr.DataArray(data_var_names, dims=[dim_name])
    da = da.assign_coords(**{dim_name: new_coord_var})

    return xr.Dataset(dict(**{var_name: da}))
