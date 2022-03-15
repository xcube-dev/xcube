#  The MIT License (MIT)
#  Copyright (c) 2022 by the xcube development team and contributors
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#
from typing import Dict, Tuple, Hashable, Optional

import xarray as xr

from xcube.util.assertions import assert_instance


def subsample_dataset(
        dataset: xr.Dataset,
        step: int,
        xy_dim_names: Optional[Tuple[str, str]] = None,
) -> xr.Dataset:
    """
    Subsample *dataset* with given integer subsampling *step*.
    Only data variables with spatial dimensions given by
    *xy_dim_names* are subsampled.

    :param dataset: the dataset providing the variables
    :param step: the integer subsampling step
    :param xy_dim_names: the spatial dimension names
    """
    assert_instance(dataset, xr.Dataset, name='dataset')
    assert_instance(step, int, name='step')
    var_slices = get_dataset_subsampling_slices(dataset,
                                                step,
                                                xy_dim_names)
    if not var_slices:
        return dataset
    return xr.Dataset(
        data_vars={
            var_name: (var[var_slices[var_name]]
                       if var_name in var_slices else var)
            for var_name, var in dataset.data_vars.items()
        },
        attrs=dataset.attrs
    )


_EMPTY_SLICE = slice(None, None, None)


def get_dataset_subsampling_slices(
        dataset: xr.Dataset,
        step: int,
        xy_dim_names: Optional[Tuple[str, str]] = None
) -> Dict[Hashable, Optional[Tuple[slice, ...]]]:
    """
    Compute subsampling slices for variables in *dataset*.
    Only data variables with spatial dimensions given by
    *xy_dim_names* are considered.

    :param dataset: the dataset providing the variables
    :param step: the integer subsampling step
    :param xy_dim_names: the spatial dimension names
    """
    assert_instance(dataset, xr.Dataset, name='dataset')
    assert_instance(step, int, name='step')
    x_dim_name, y_dim_name = xy_dim_names or ('x', 'y')
    slices_dict: Dict[Tuple[Hashable, ...], Tuple[slice, ...]] = dict()
    vars_dict: Dict[Hashable, Optional[Tuple[slice, ...]]] = dict()
    for var_name, var in dataset.data_vars.items():
        var_index = slices_dict.get(var.dims)
        if var_index is None:
            var_index = None
            for index, dim_name in enumerate(var.dims):
                if dim_name == x_dim_name or dim_name == y_dim_name:
                    if var_index is None:
                        var_index = index * [_EMPTY_SLICE]
                    var_index.append(slice(None, None, step))
                elif var_index is not None:
                    var_index.append(_EMPTY_SLICE)
            if var_index is not None:
                var_index = tuple(var_index)
                slices_dict[var.dims] = tuple(var_index)
        if var_index is not None:
            vars_dict[var_name] = var_index
    return vars_dict
