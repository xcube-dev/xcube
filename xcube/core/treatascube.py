# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
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

from typing import Mapping

import xarray as xr
from xcube.core.gridmapping import GridMapping
from xcube.core.normalize import normalize_dataset
from xcube.core.timecoord import get_time_range_from_data
from xcube.core.verify import assert_cube


def verify_cube_subset(dataset: xr.Dataset):
    """
    Verifies that the dataset fulfils the minimum requirements  for a dataset
    that either is or may be converted to be a cube. In order to do so, the
    dataset
     * must have two spatial dimensions
     * must have at least one data variable that uses the spatial dimensions
     * must have either a temporal dimension or temporal information in its
     attributes

    :param dataset: The dataset to be validated.
    :raise: ValueError, if dataset contains no subset that is a valid xcube
    dataset.
    """
    grid_mapping = GridMapping.from_dataset(dataset)
    # if a gridmapping exists, the dataset contains spatial dimensions
    # if no gridmapping exists, a ValueError is raised
    at_least_one_valid_var = False
    for data_var in dataset.data_vars.values():
        if grid_mapping.xy_dim_names[0] in data_var.dims and \
                grid_mapping.xy_dim_names[1] in data_var.dims:
            at_least_one_valid_var = True
            break
    if not at_least_one_valid_var:
        raise ValueError('Not at least one data variable has '
                         'spatial dimensions.')
    start_time, end_time = get_time_range_from_data(dataset)
    if start_time is None and end_time is None:
        raise ValueError('Dataset has no temporal information.')


def split_cube(dataset: xr.Dataset) -> (xr.Dataset, Mapping[str, xr.DataArray]):
    """
    Creates a subset of a dataset that meets all hard requirements of a cube.
    To this end, all variables that do not include spatial dimensions will be
    removed and returned in a mapping from dataset name to data array.

    :param dataset: The dataset from which the subset shall be built
    :raise: ValueError, , if dataset contains no subset that is a valid xcube
    dataset.
    :return: a tuple, consisting of (a) a subset of the input dataset that has
    been normalized to conform to strict cube requirements and (b) a mapping
    of the names of removed data variables to these data variables
    """
    verify_cube_subset(dataset)

    non_cube_data_vars = dict()
    grid_mapping = GridMapping.from_dataset(dataset)

    for data_var_name, data_var in dataset.data_vars.items():
        if grid_mapping.xy_dim_names[0] not in data_var.dims \
                and grid_mapping.xy_dim_names[1] not in data_var.dims:
            non_cube_data_vars[data_var_name] = data_var
    dataset = dataset.drop_vars(list(non_cube_data_vars.keys()))
    dataset = normalize_dataset(dataset, do_not_normalize_spatial_dims=True)
    return dataset, non_cube_data_vars


def merge_cube(dataset: xr.Dataset,
               data_vars: Mapping[str, xr.DataArray]) -> xr.Dataset:
    """
    Merges data_vars into a data set.

    :param dataset: The dataset into which the data variables shall be merged
    :param data_vars: The data variables that shall be merged into the dataset
    :raise: ValueError, if dataset is not a valid xcube dataset
    """
    assert_cube(dataset)
    return dataset.assign(data_vars)
