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
from .request import CubeGeneratorRequestLike
from .response import CubeInfo, CubeGeneratorResult
from .remote.config import ServiceConfigLike


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
        generator remote, otherwise a local cube generator is configured
        using optional *stores_config*.

        The *service_config* parameter can be passed in different ways:

        * An instance of :class:ServiceConfig.
        * A ``str``. Then it is interpreted as a path to a YAML or JSON file
          and the remote configuration is loaded from this file.
          The file content may include template variables that are
          interpolated by environment variables,
          e.g. "${XCUBE_GEN_CLIENT_SECRET}".
        * A ``dict``. Then it is interpreted as a remote configuration
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
            from .remote.config import ServiceConfig
            from .remote.generator import RemoteCubeGenerator
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
            from .local.generator import LocalCubeGenerator
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
    def generate_cube(self, request: CubeGeneratorRequestLike) \
            -> CubeGeneratorResult:
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
