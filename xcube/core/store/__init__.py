# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from .accessor import (
    DataDeleter,
    DataOpener,
    DataPreloader,
    DataTimeSliceUpdater,
    DataWriter,
    find_data_opener_extensions,
    find_data_writer_extensions,
    get_data_accessor_predicate,
    new_data_opener,
    new_data_writer,
)
from .assertions import assert_valid_config, assert_valid_params
from .datatype import (
    ANY_TYPE,
    DATASET_TYPE,
    GEO_DATA_FRAME_TYPE,
    MULTI_LEVEL_DATASET_TYPE,
    DataType,
    DataTypeLike,
)
from .descriptor import (
    DataDescriptor,
    DatasetDescriptor,
    GeoDataFrameDescriptor,
    MultiLevelDatasetDescriptor,
    VariableDescriptor,
    new_data_descriptor,
)
from .error import DataStoreError
from .fs.registry import get_filename_extensions, new_fs_data_store
from .preload import PreloadHandle, PreloadState, PreloadStatus
from .search import DataSearcher, DefaultSearchMixin
from .store import (
    DataStore,
    MutableDataStore,
    PreloadedDataStore,
    find_data_store_extensions,
    get_data_store_class,
    get_data_store_params_schema,
    list_data_store_ids,
    new_data_store,
)
from .storepool import (
    DataStoreConfig,
    DataStoreInstance,
    DataStorePool,
    DataStorePoolLike,
    get_data_store_instance,
)

__all__ = [
    "ANY_TYPE",
    "DATASET_TYPE",
    "GEO_DATA_FRAME_TYPE",
    "MULTI_LEVEL_DATASET_TYPE",
    "DataDeleter",
    "DataDescriptor",
    "DataOpener",
    "DataPreloader",
    "DataSearcher",
    "DataStore",
    "DataStoreConfig",
    "DataStoreError",
    "DataStoreInstance",
    "DataStorePool",
    "DataStorePoolLike",
    "DataTimeSliceUpdater",
    "DataType",
    "DataTypeLike",
    "DataWriter",
    "DatasetDescriptor",
    "DefaultSearchMixin",
    "GeoDataFrameDescriptor",
    "MultiLevelDatasetDescriptor",
    "MutableDataStore",
    "PreloadHandle",
    "PreloadState",
    "PreloadStatus",
    "PreloadedDataStore",
    "VariableDescriptor",
    "assert_valid_config",
    "assert_valid_params",
    "find_data_opener_extensions",
    "find_data_store_extensions",
    "find_data_writer_extensions",
    "get_data_accessor_predicate",
    "get_data_store_class",
    "get_data_store_instance",
    "get_data_store_params_schema",
    "get_filename_extensions",
    "list_data_store_ids",
    "new_data_descriptor",
    "new_data_opener",
    "new_data_store",
    "new_data_writer",
    "new_fs_data_store",
]
