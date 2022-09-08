# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import collections.abc
from typing import Union, Iterator

import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from .config import S3_MODE_ML_DATASET
from ..datasets.context import DatasetsContext
from ...util.assertions import assert_instance


class DatasetsMapping(collections.abc.Mapping):
    """Represents the given *datasets_ctx* as a mapping from
    dataset identifier to dataset, it can
    be passed to class:EmulatedObjectStorage.

    This is the applied Adapter design pattern to make
    class:DatasetsContext compatible with the mapping argument for
    class:EmulatedObjectStorage.

    :param datasets_ctx: The datasets context
    """

    def __init__(self,
                 datasets_ctx: DatasetsContext,
                 s3_mode: str):
        assert_instance(datasets_ctx, DatasetsContext, name="datasets_ctx")
        assert_instance(s3_mode, str, name="s3_mode")
        self._datasets_ctx = datasets_ctx
        self._s3_mode = s3_mode

    def __len__(self) -> int:
        return len(self._datasets_ctx.get_dataset_configs())

    def __iter__(self) -> Iterator[str]:
        return iter(c["Identifier"]
                    for c in self._datasets_ctx.get_dataset_configs())

    def __contains__(self, dataset_id: str) -> bool:
        """Check if *dataset_id* is a valid dataset.
        Overridden to avoid a call to __getitem__(),
        which will open the dataset (or raise ApiError!),
        but we want this to happen for direct __getitem__()
        calls only."""
        dataset_configs = self._datasets_ctx.get_dataset_configs()
        dataset_config = self._datasets_ctx.find_dataset_config(
            dataset_configs, dataset_id
        )
        return dataset_config is not None

    def __getitem__(self, dataset_id: str) \
            -> Union[xr.Dataset, MultiLevelDataset]:
        """Get or open the dataset given by *dataset_id*."""
        # Will raise ApiError
        if self._s3_mode == S3_MODE_ML_DATASET:
            return self._datasets_ctx.get_ml_dataset(dataset_id)
        else:
            return self._datasets_ctx.get_dataset(dataset_id)
