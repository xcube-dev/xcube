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

from typing import Sequence

from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataStoreError
from xcube.core.store import DataStorePool
from xcube.core.store import DatasetDescriptor
from xcube.core.store import get_data_store_instance
from xcube.core.store import new_data_opener
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.progress import observe_progress
from ..config import InputConfig
from ..error import CubeGeneratorError


class DatasetsDescriber:

    def __init__(self,
                 input_configs: Sequence[InputConfig],
                 store_pool: DataStorePool = None):
        assert_true(len(input_configs) > 0,
                    'At least one input must be given')
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, 'store_pool')
        self._input_configs = input_configs
        self._store_pool = store_pool

    def describe_datasets(self) -> Sequence[DatasetDescriptor]:
        descriptors = []
        with observe_progress('Fetching dataset information',
                              len(self._input_configs)) as progress:
            for input_config in self._input_configs:
                descriptors.append(self._describe_dataset(input_config))
                progress.worked(1)
        return descriptors

    def _describe_dataset(self, input_config: InputConfig) \
            -> DatasetDescriptor:
        opener_id = input_config.opener_id
        store_params = input_config.store_params or {}
        if input_config.store_id:
            store_instance = get_data_store_instance(
                input_config.store_id,
                store_params=store_params,
                store_pool=self._store_pool
            )
            opener = store_instance.store
        else:
            opener = new_data_opener(opener_id)
        try:
            descriptor = opener.describe_data(input_config.data_id,
                                              data_type=DATASET_TYPE)
        except DataStoreError as dse:
            raise CubeGeneratorError(f'{dse}',
                                     status_code=400) from dse
        if not isinstance(descriptor, DatasetDescriptor):
            raise RuntimeError(f'internal error: data store '
                               f'"{input_config.store_id}": '
                               f'expected DatasetDescriptor but got '
                               f'a {type(descriptor)}')
        return descriptor
