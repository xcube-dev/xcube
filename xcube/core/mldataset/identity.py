# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from .abc import MultiLevelDataset
from .mapped import MappedMultiLevelDataset


class IdentityMultiLevelDataset(MappedMultiLevelDataset):
    """The identity."""

    def __init__(self, ml_dataset: MultiLevelDataset, ds_id: str = None):
        super().__init__(ml_dataset, lambda ds: ds, ds_id=ds_id)
