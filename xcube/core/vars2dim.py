# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

import xarray as xr

from xcube.core.verify import assert_cube


def vars_to_dim(cube: xr.Dataset,
                dim_name: str = 'var',
                var_name='data',
                cube_asserted: bool = False):
    """
    Convert data variables into a dimension.

    :param cube: The xcube dataset.
    :param dim_name: The name of the new dimension and coordinate variable. Defaults to 'var'.
    :param var_name: The name of the new, single data variable. Defaults to 'data'.
    :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
    :return: A new xcube dataset with data variables turned into a new dimension.
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
