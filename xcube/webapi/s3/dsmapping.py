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
from ..datasets.context import DatasetsContext
from ...util.assertions import assert_instance


class DatasetsMapping(collections.abc.Mapping):
    """Represents the given *datasets_ctx* as a mapping from
    dataset identifier to dataset, it can
    be passed to class:EmulatedObjectStorage.

    This is the applied Adapter design pattern to make
    class:DatasetsContext compatible with the mapping argument for
    class:EmulatedObjectStorage.

    :param datasets_ctx: The datasets' context
    :param is_multi_level: Whether this is a multi-level datasets
        object storage
    """

    def __init__(self,
                 datasets_ctx: DatasetsContext,
                 is_multi_level: bool = False):
        assert_instance(datasets_ctx, DatasetsContext, name="datasets_ctx")
        assert_instance(is_multi_level, bool, name="is_multi_level")
        self._datasets_ctx = datasets_ctx
        self._is_multi_level = is_multi_level
        self._s3_names = self._get_s3_names(datasets_ctx, is_multi_level)

    @staticmethod
    def _get_s3_names(datasets_ctx: DatasetsContext,
                      is_multi_level: bool):
        s3_names = {}
        for c in datasets_ctx.get_dataset_configs():
            ds_id: str = c["Identifier"]
            s3_name = ds_id
            if is_multi_level:
                if not s3_name.endswith(".levels"):
                    s3_name += ".levels"
            else:
                if not s3_name.endswith(".zarr"):
                    s3_name += ".zarr"
            s3_names[s3_name] = ds_id
        return s3_names

    def __len__(self) -> int:
        return len(self._s3_names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._s3_names)

    def __contains__(self, s3_name: str) -> bool:
        """Check if *dataset_id* is a valid dataset.
        Overridden to avoid a call to __getitem__(),
        which will open the dataset (or raise ApiError!),
        but we want this to happen for direct __getitem__()
        calls only."""
        return s3_name in self._s3_names

    def __getitem__(self, s3_name: str) \
            -> Union[xr.Dataset, MultiLevelDataset]:
        """Get or open the dataset given by *dataset_id*."""
        dataset_id = self._s3_names[s3_name]
        # Will raise ApiError
        if self._is_multi_level:
            return self._datasets_ctx.get_ml_dataset(dataset_id)
        else:
            return self._datasets_ctx.get_dataset(dataset_id)
