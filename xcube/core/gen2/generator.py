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
from abc import ABC, abstractmethod
from typing import Optional

from xcube.core.store import DataStorePool
from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .combiner import CubesCombiner
from .informant import CubeInformant
from .opener import CubesOpener
from .progress import ApiProgressCallbackObserver
from .progress import ConsoleProgressObserver
from .request import CubeGeneratorRequest
from .response import CubeInfo
from .writer import CubeWriter


class CubeGenerator(ABC):
    @classmethod
    def from_file(cls,
                  gen_config_path: Optional[str],
                  stores_config_path: str = None,
                  service_config_path: str = None,
                  verbosity: int = 0) -> 'CubeGenerator':
        """
        Create a cube generator from configuration files.

        *gen_config_path* is the cube generator configuration. It may be
        provided as a JSON or YAML file (file extensions ".json" or ".yaml").
        If the *gen_config_path* argument is omitted, it is expected that
        the cube generator configuration is piped as a JSON string.

        *stores_config_path* is a path to a JSON file with data store
        configurations. It is a mapping of names to
        configured stores. Entries are dictionaries that have a mandatory
        "store_id" property which is a name of a registered xcube data store.
        The optional "store_params" property may define data store specific
        parameters.

        *stores_config_path* and *service_config_path* cannot be given
        at the same time.

        :param gen_config_path: Cube generation configuration. It may be
            provided as a JSON or YAML file (file extensions ".json" or ".yaml").
            If None is passed, it is expected that
            the cube generator configuration is piped as a JSON string.
        :param stores_config_path: A path to a JSON or YAML file that represents
            mapping of store names to rized stores.
        :param service_config_path: A path to a JSON or YAML file that configures an
            xcube generator service.
        :param verbosity: Level of verbosity, 0 means off.
        """
        assert_instance(gen_config_path, (str, type(None)), 'gen_config_path')
        assert_instance(stores_config_path, (str, type(None)), 'stores_config_path')
        assert_instance(service_config_path, (str, type(None)), 'service_config_path')
        assert_condition(not (stores_config_path is not None and
                              service_config_path is not None),
                         'stores_config_path and service_config_path cannot be'
                         ' given at the same time.')

        request = CubeGeneratorRequest.from_file(gen_config_path, verbosity=verbosity)

        if service_config_path is not None:
            from .service import ServiceConfig
            from .service import CubeGeneratorService
            service_config = ServiceConfig.from_file(service_config_path) \
                if service_config_path is not None else None
            return CubeGeneratorService(request,
                                        service_config=service_config,
                                        verbosity=verbosity)
        else:
            store_pool = DataStorePool.from_file(stores_config_path) \
                if stores_config_path is not None else None
            return LocalCubeGenerator(request,
                                      store_pool=store_pool,
                                      verbosity=verbosity)

    @abstractmethod
    def get_cube_info(self) -> CubeInfo:
        """Get data cube information."""

    @abstractmethod
    def generate_cube(self):
        """Generate a data cube."""


class LocalCubeGenerator(CubeGenerator):
    """
    Generator tool for data cubes.

    Creates cube views from one or more cube stores, resamples them to a
    common grid, optionally performs some cube transformation, and writes
    the resulting cube to some target cube store.

    :param request: Cube generation request.
    :param store_pool: An optional pool of pre-configured data stores
        referenced from *gen_config* input/output configurations.
    :param verbosity: Level of verbosity, 0 means off.
    """

    def __init__(self,
                 request: CubeGeneratorRequest,
                 store_pool: DataStorePool = None,
                 verbosity: int = False):
        assert_instance(request, CubeGeneratorRequest, 'request')
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, 'store_pool')

        self._request = request
        self._store_pool = store_pool if store_pool is not None \
            else DataStorePool()
        self._verbosity = verbosity

    def generate_cube(self):
        request = self._request

        if request.callback_config:
            ApiProgressCallbackObserver(request.callback_config).activate()

        if self._verbosity:
            ConsoleProgressObserver().activate()

        cubes_opener = CubesOpener(request.input_configs,
                                   request.cube_config,
                                   store_pool=self._store_pool)

        cube_combiner = CubesCombiner(request.cube_config)

        cube_writer = CubeWriter(request.output_config,
                                 store_pool=self._store_pool)

        with observe_progress('Generating cube', 100) as cm:
            cm.will_work(10)
            cubes = cubes_opener.open_cubes()

            cm.will_work(10)
            cube = cube_combiner.process_cubes(cubes)

            cm.will_work(80)
            data_id = cube_writer.write_cube(cube)

        if self._verbosity:
            print('Cube "{}" generated within {:.2f} seconds'
                  .format(str(data_id), cm.state.total_time))

    def get_cube_info(self) -> CubeInfo:
        informant = CubeInformant(request=self._request, store_pool=self._store_pool)
        return informant.generate()
