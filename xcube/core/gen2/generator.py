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
from typing import Any, Optional

from xcube.core.store import DataStorePool
from xcube.core.store import DataStorePoolLike
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.progress import observe_progress
from .codeexec import CubeCodeExecutor
from .combiner import CubesCombiner
from .informant import CubeInformant
from .opener import CubesOpener
from .processor import NoOpCubeProcessor
from .progress import ApiProgressCallbackObserver
from .progress import ConsoleProgressObserver
from .request import CubeGeneratorRequest
from .request import CubeGeneratorRequestLike
from .response import CubeInfo
from .service.config import ServiceConfigLike
from .writer import CubeWriter

_CLASS_METHOD_NAME_PROCESS_DATASET = 'process_dataset'
_CLASS_METHOD_NAME_GET_PARAMS_SCHEMA = 'get_params_schema'


class CubeGenerator(ABC):
    """
    Abstract base class for cube generators.

    Use the ``CubeGenerator.load()`` method to instantiate new
    cube generators.
    """

    @classmethod
    def new(cls,
            service_config: Optional[ServiceConfigLike] = None,
            stores_config: Optional[DataStorePoolLike] = None,
            verbosity: int = 0,
            **kwargs) -> 'CubeGenerator':
        """
        Create a new cube generator from given configurations.

        If *service_config* is given, it describes a remote xcube
        generator service, otherwise a local cube generator is configured
        using optional *stores_config*.

        The *service_config* parameter can be passed in different ways:

        * An instance of :class:ServiceConfig.
        * A ``str``. Then it is interpreted as a path to a YAML or JSON file
          and the service configuration is loaded from this file.
          The file content may include template variables that are interpolated
          by environment variables, e.g. "${XCUBE_GEN_CLIENT_SECRET}".
        * A ``dict``. Then it is interpreted as a service configuration
          JSON object.

        If *stores_config* is given, it describes a pool of data stores to be
        used as input and output for the cube generator. *stores_config*
        if a mapping of store instance identifiers to configured store
        instances. A store instance is a dictionary that has a mandatory
        "store_id" property which is a name of a registered xcube data store.
        as well as an optional "store_params" property that may define data
        store specific parameters.

        Similar to *service_config*, the *stores_config* parameter
        can be passed in different ways:

        * An instance of :class:DataStorePool.
        * A ``str``. Then it is interpreted as a YAML or JSON file path
          and the stores configuration is loaded from this file.
        * A ``dict``. Then it is interpreted as a stores configuration
          JSON object.

        The *service_config* and *stores_config* parameters cannot
        be given both.

        :param service_config: Service configuration.
        :param stores_config: Data stores configuration.
        :param verbosity: Level of verbosity, 0 means off.
        :param kwargs: Extra arguments passed to the generator constructors.
        """
        if service_config is not None:
            from .service.config import ServiceConfig
            from .service.generator import RemoteCubeGenerator
            assert_true(stores_config is None,
                        'service_config and stores_config cannot be'
                        ' given at the same time.')
            assert_instance(service_config,
                            (str, dict, ServiceConfig, type(None)),
                            'service_config')
            service_config = ServiceConfig.normalize(service_config) \
                if service_config is not None else None
            return RemoteCubeGenerator(service_config=service_config,
                                       verbosity=verbosity,
                                       **kwargs)
        else:
            assert_instance(stores_config,
                            (str, dict, DataStorePool, type(None)),
                            'stores_config')
            store_pool = DataStorePool.normalize(stores_config) \
                if stores_config is not None else None
            return LocalCubeGenerator(store_pool=store_pool,
                                      verbosity=verbosity)

    @abstractmethod
    def get_cube_info(self, request: CubeGeneratorRequestLike) -> CubeInfo:
        """
        Get data cube information for given *request*.

        The *request* argument can be
        * an instance of ``CubeGeneratorRequest``;
        * a ``dict``. In this case it is interpreted as JSON object and
          parsed into a ``CubeGeneratorRequest``;
        * a ``str``. In this case it is interpreted as path to a
          YAML or JSON file, which is loaded and
          parsed into a ``CubeGeneratorRequest``.

        :param request: Cube generator request.
        :return: a cube information object
        :raises CubeGeneratorError: if cube info generation failed
        :raises DataStoreError: if data store access failed
        """

    @abstractmethod
    def generate_cube(self, request: CubeGeneratorRequestLike) -> Any:
        """
        Generate the data cube for given *request*.

        The *request* argument can be
        * an instance of ``CubeGeneratorRequest``;
        * a ``dict``. In this case it is interpreted as JSON object and
          parsed into a ``CubeGeneratorRequest``;
        * a ``str``. In this case it is interpreted as path to a
          YAML or JSON file, which is loaded and
          parsed into a ``CubeGeneratorRequest``.

        Returns the cube reference which can be used as ``data_id`` in
        ``store.open_data(data_id)`` where *store*  refers to the
        store configured in ``output_config`` of the cube generator request.

        :param request: Cube generator request.
        :return: the cube reference
        :raises CubeGeneratorError: if cube generation failed
        :raises DataStoreError: if data store access failed
        """


class LocalCubeGenerator(CubeGenerator):
    """
    Generator tool for data cubes.

    Creates cube views from one or more cube stores, resamples them to a
    common grid, optionally performs some cube transformation, and writes
    the resulting cube to some target cube store.

    :param store_pool: An optional pool of pre-configured data stores
        referenced from *gen_config* input/output configurations.
    :param verbosity: Level of verbosity, 0 means off.
    """

    def __init__(self,
                 store_pool: DataStorePool = None,
                 verbosity: int = 0):
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, 'store_pool')

        self._store_pool = store_pool if store_pool is not None \
            else DataStorePool()
        self._verbosity = verbosity

    def generate_cube(self, request: CubeGeneratorRequestLike) -> Any:
        request = CubeGeneratorRequest.normalize(request)
        request = request.for_local()

        # noinspection PyUnusedLocal
        def _no_op_callable(ds, **kwargs):
            return ds

        if request.code_config is not None:
            code_executor = CubeCodeExecutor(request.code_config)
        else:
            code_executor = NoOpCubeProcessor()

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
            cube = code_executor.process_cube(cube)

            cm.will_work(80)
            data_id = cube_writer.write_cube(cube)

        if self._verbosity:
            print('Cube "{}" generated within {:.2f} seconds'
                  .format(str(data_id), cm.state.total_time))

        return data_id

    def get_cube_info(self, request: CubeGeneratorRequestLike) -> CubeInfo:
        request = CubeGeneratorRequest.normalize(request)
        informant = CubeInformant(request=request.for_local(),
                                  store_pool=self._store_pool)
        return informant.generate()
