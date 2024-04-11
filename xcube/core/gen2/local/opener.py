# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import traceback

from xcube.core.normalize import DatasetIsNotACubeError
from xcube.core.normalize import decode_cube
from xcube.core.store import DATASET_TYPE
from xcube.core.store import DataStoreError
from xcube.core.store import DataStorePool
from xcube.core.store import get_data_store_instance
from xcube.core.store import new_data_opener
from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .transformer import TransformedCube
from ..config import CubeConfig
from ..config import InputConfig
from ..error import CubeGeneratorError

# Names of cube configuration parameters that
# are not shared with open parameters.
_STEADY_CUBE_CONFIG_NAMES = {"chunks", "tile_size"}


class CubeOpener:
    def __init__(self, cube_config: CubeConfig, store_pool: DataStorePool = None):
        assert_instance(cube_config, CubeConfig, "cube_config")
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, "store_pool")
        self._cube_config = cube_config
        self._store_pool = store_pool

    def open_cube(self, input_config: InputConfig) -> TransformedCube:
        cube_config = self._cube_config
        cube_params = cube_config.to_dict()
        opener_id = input_config.opener_id
        store_params = input_config.store_params or {}
        open_params = input_config.open_params or {}

        with observe_progress("reading cube", 3) as observer:
            try:
                if input_config.store_id:
                    store_instance = get_data_store_instance(
                        input_config.store_id,
                        store_params=store_params,
                        store_pool=self._store_pool,
                    )
                    store = store_instance.store
                    if opener_id is None:
                        opener_id = self._get_opener_id(input_config, store)
                    opener = store
                    open_params = dict(open_params)
                    open_params["opener_id"] = opener_id
                else:
                    opener = new_data_opener(opener_id)
                    open_params = dict(open_params)
                    open_params.update(store_params)

                open_params_schema = opener.get_open_data_params_schema(
                    input_config.data_id
                )

                dataset_open_params = {
                    k: v
                    for k, v in cube_params.items()
                    if k in open_params_schema.properties
                }

                observer.worked(1)

                dataset = opener.open_data(
                    input_config.data_id, **open_params, **dataset_open_params
                )
                observer.worked(1)

            except DataStoreError as dse:
                raise CubeGeneratorError(f"{dse}", status_code=400) from dse

            # Turn dataset into cube and grid_mapping
            try:
                cube, gm, _ = decode_cube(dataset, normalize=True)
            except DatasetIsNotACubeError as e:
                raise CubeGeneratorError(f"{e}") from e
            observer.worked(1)

        if dataset_open_params:
            drop_names = [
                k
                for k in dataset_open_params.keys()
                if k not in _STEADY_CUBE_CONFIG_NAMES
            ]
            cube_config = cube_config.drop_props(drop_names)

        return cube, gm, cube_config

    @classmethod
    def _get_opener_id(cls, input_config, store) -> str:
        opener_ids = None
        data_type_names = store.get_data_types_for_data(input_config.data_id)
        for data_type_name in data_type_names:
            if DATASET_TYPE.is_super_type_of(data_type_name):
                opener_ids = store.get_data_opener_ids(
                    data_id=input_config.data_id, data_type=data_type_name
                )
                break
        if not opener_ids:
            raise CubeGeneratorError(
                f"Data store {input_config.store_id!r}" f" does not support datasets",
                status_code=400,
            )
        opener_id = opener_ids[0]
        return opener_id
