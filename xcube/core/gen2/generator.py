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
from typing import Optional

import xarray as xr

from xcube.core.store import DataStorePool
from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .combiner import CubesCombiner
from .config import CubeGeneratorConfig
from .opener import CubesOpener
from .progress import ApiProgressCallbackObserver
from .progress import ConsoleProgressObserver
from .writer import CubeWriter


class CubeGenerator:
    """
    Generator tool for data cubes.

    Creates cube views from one or more cube stores, resamples them to a
    common grid, optionally performs some cube transformation, and writes
    the resulting cube to some target cube store.

    :param gen_config: Cube generation configuration.
    :param store_pool: An optional pool of pre-configured data stores
        referenced from *gen_config* input/output configurations.
    :param verbose: Whether to output progress information to stdout.
    """

    def __init__(self,
                 gen_config: CubeGeneratorConfig,
                 store_pool: DataStorePool = None,
                 verbose: bool = False):
        assert_instance(gen_config, CubeGeneratorConfig, 'gen_config')
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, 'store_pool')

        self._gen_config = gen_config
        self._store_pool = store_pool if store_pool is not None \
            else DataStorePool()
        self._verbose = verbose

    @classmethod
    def from_file(cls,
                  gen_config_path: Optional[str],
                  store_configs_path: str = None,
                  verbose: bool = False) -> 'CubeGenerator':
        """
        Create a cube generator from configuration files.

        *gen_config_path* is the cube generator configuration. It may be
        provided as a JSON or YAML file (file extensions ".json" or ".yaml").
        If the *gen_config_path* argument is omitted, it is expected that
        the cube generator configuration is piped as a JSON string.

        *store_configs_path* is a path to a JSON file with data store
        configurations. It is a mapping of names to
        configured stores. Entries are dictionaries that have a mandatory
        "store_id" property which is a name of a registered xcube data store.
        The optional "store_params" property may define data store specific
        parameters.

        :param gen_config_path: Cube generation configuration. It may be
            provided as a JSON or YAML file (file extensions ".json" or ".yaml").
            If None is passed, it is expected that
            the cube generator configuration is piped as a JSON string.
        :param store_configs_path: A JSON file that maps store names to
            parameterized stores.
        :param verbose: Whether to output progress information to stdout.
        """
        gen_config = CubeGeneratorConfig.from_file(gen_config_path, verbose=verbose)
        store_pool = DataStorePool.from_file(store_configs_path) \
            if store_configs_path else None
        return CubeGenerator(gen_config, store_pool, verbose=verbose)

    def run(self) -> xr.Dataset:
        gen_config = self._gen_config

        if gen_config.callback_config:
            ApiProgressCallbackObserver(gen_config.callback_config).activate()

        if self._verbose:
            ConsoleProgressObserver().activate()

        cubes_opener = CubesOpener(gen_config.input_configs,
                                   gen_config.cube_config,
                                   store_pool=self._store_pool)

        cube_combiner = CubesCombiner(gen_config.cube_config)

        cube_writer = CubeWriter(gen_config.output_config,
                                 store_pool=self._store_pool)

        with observe_progress('Generating cube', 100) as cm:
            cm.will_work(10)
            cubes = cubes_opener.open_cubes()

            cm.will_work(10)
            cube = cube_combiner.process_cubes(cubes)

            cm.will_work(80)
            data_id = cube_writer.write_cube(cube)

        if self._verbose:
            print('Cube "{}" generated within {:.2f} seconds'
                  .format(str(data_id), cm.state.total_time))

        return cube
