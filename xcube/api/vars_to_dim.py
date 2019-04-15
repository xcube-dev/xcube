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

def vars_to_dim(cube: xr.Dataset, dim_name: str = 'newdim'):
    """
    Convert data variables into a dimension

    :param cube: The data cube.
    :param dim_name: The name of the new dimension ['vars']
    :return: A new data cube with the new dimension.
    """

    if len(cube.variables) == 0:
        raise ValueError("ERROR: Cube has no variables. Exiting.")

    da = xr.concat([cube[var_name] for var_name in cube.data_vars], dim_name)
    var_coords = xr.DataArray([var_name for var_name in cube.data_vars], dims=[dim_name])
    da = da.assign_coords(**{dim_name: var_coords})

    return xr.Dataset(dict(**{dim_name + '_vars': da}))
