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

from .accessor import DataDeleter
from .accessor import DataOpener
from .accessor import DataTimeSliceUpdater
from .accessor import DataWriter
from .accessor import find_data_opener_extensions
from .accessor import find_data_writer_extensions
from .accessor import get_data_accessor_predicate
from .accessor import new_data_opener
from .accessor import new_data_writer
from .assertions import assert_valid_config
from .assertions import assert_valid_params
from .datatype import ANY_TYPE
from .datatype import DATASET_TYPE
from .datatype import DataType
from .datatype import DataTypeLike
from .datatype import GEO_DATA_FRAME_TYPE
from .datatype import MULTI_LEVEL_DATASET_TYPE
from .descriptor import DataDescriptor
from .descriptor import DatasetDescriptor
from .descriptor import GeoDataFrameDescriptor
from .descriptor import MultiLevelDatasetDescriptor
from .descriptor import VariableDescriptor
from .descriptor import new_data_descriptor
from .error import DataStoreError
from .fs.registry import new_fs_data_store
from .search import DataSearcher
from .search import DefaultSearchMixin
from .store import DataStore
from .store import MutableDataStore
from .store import find_data_store_extensions
from .store import get_data_store_class
from .store import get_data_store_params_schema
from .store import new_data_store
from .storepool import DataStoreConfig
from .storepool import DataStoreInstance
from .storepool import DataStorePool
from .storepool import DataStorePoolLike
from .storepool import get_data_store_instance
