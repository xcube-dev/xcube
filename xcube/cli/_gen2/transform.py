# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from xcube.core.store.param import ParamDescriptor
from xcube.core.store.param import ParamDescriptorSet
from xcube.core.store.param import ParamValues

# Need to be aligned with params in transform_cube(cube, **params)
TRANSFORM_PARAMS = ParamDescriptorSet([
    ParamDescriptor('python_code', dtype=str),
    ParamDescriptor('gh_repo_name', dtype=str),
    ParamDescriptor('gh_user_name', dtype=str),
    ParamDescriptor('gh_access_token', dtype=str),
    ParamDescriptor('transform_name', dtype=str),
    ParamDescriptor('transform_params', dtype=dict),
])


def transform_cube(cube: xr.Dataset,
                   python_code: str = None,
                   gh_repo_name: str = None,
                   gh_user_name: str = None,
                   gh_access_token: str = None,
                   transform_name: str = None,
                   transform_params: ParamValues = None):
    """Use user-defined Python code to transform *cube*."""
    # TODO: implement me
    return cube
