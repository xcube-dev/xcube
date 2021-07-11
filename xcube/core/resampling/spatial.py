# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

from typing import Union, Callable

import numpy as np
import xarray as xr
from dask import array as da

from xcube.core.gridmapping import GridMapping
from .affine import affine_transform_dataset
from .rectify import rectify_dataset

NDImage = Union[np.ndarray, da.Array]
Aggregator = Callable[[NDImage], NDImage]


def resample_in_space(dataset: xr.Dataset,
                      source_gm: GridMapping = None,
                      target_gm: GridMapping = None):
    """
    Resample a dataset in the spatial dimensions.

    If the source grid mapping *source_gm* is not given,
    it is derived from *dataset*:
    ``source_gm = GridMapping.from_dataset(dataset)``.

    If the target grid mapping *target_gm* is not given,
    it is derived from *source_gm*:
    ``target_gm = source_gm.to_regular()``.

    If *source_gm* is almost equal to *target_gm*, this
    function is a no-op and *dataset* is returned unchanged.

    Otherwise the function computes a spatially
    resampled version of *dataset* and returns it.

    :param dataset: The source dataset.
    :param source_gm: The source grid mapping.
    :param target_gm: The target grid mapping. Must be regular.

    :return: The spatially resampled dataset.
    """
    if source_gm is None:
        # No source grid mapping given, so do derive it from dataset
        source_gm = GridMapping.from_dataset(dataset)

    if target_gm is None:
        # No target grid mapping given, so do derive it from source
        target_gm = source_gm.to_regular()

    if source_gm.is_close(target_gm):
        # If source and target grid mappings are almost equal
        return dataset

    # target_gm must be regular
    GridMapping.assert_regular(target_gm, name='target_gm')

    # Are source and target both geographic grid mappings?
    both_geographic = source_gm.crs.is_geographic \
                      and target_gm.crs.is_geographic

    if not both_geographic \
            and source_gm.crs != target_gm.crs:
        # If CRSes are not both geographic and their CRSes are different
        # transform the source_gm so its CRS matches the target CRS:
        transformed_geo_coding = source_gm.transform(target_gm.crs)
        return resample_in_space(dataset,
                                 source_gm=transformed_geo_coding,
                                 target_gm=target_gm)
    else:
        # If CRSes are both geographic or their CRSes are equal:
        if source_gm.is_regular:
            # If also the source is regular, then resampling reduces
            # to an affine transformation.
            return affine_transform_dataset(dataset,
                                            source_gm=source_gm,
                                            target_gm=target_gm)
        else:
            # If the source is not regular, we need to rectify it,
            # so the target is regular. Our rectification implementations
            # works only if source pixel size > target pixel size.
            # Therefore check if we must downscale source first.
            x_scale = source_gm.x_res / target_gm.x_res
            y_scale = source_gm.y_res / target_gm.y_res
            xy_scale = 0.5 * (x_scale + y_scale)  # avg scale
            if xy_scale > 0.95:  # magic constant!
                # Source has lower resolution than target.
                return rectify_dataset(dataset,
                                       source_gm=source_gm,
                                       target_gm=target_gm)
            else:
                # Source has higher resolution than target.
                # Downscale first, then rectify
                downscaled_dataset = affine_transform_dataset(
                    dataset,
                    source_gm=source_gm,
                    target_gm=target_gm
                )
                x_name, y_name = source_gm.xy_var_names
                downscaled_gm = GridMapping.from_coords(
                    downscaled_dataset[x_name],
                    downscaled_dataset[y_name],
                    source_gm.crs
                )
                return rectify_dataset(downscaled_dataset,
                                       source_gm=downscaled_gm,
                                       target_gm=target_gm)
