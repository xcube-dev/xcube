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

from typing import Any

from xcube.core.store import DataStorePool
from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .combiner import CubesCombiner
from .informant import CubeInformant
from .opener import CubesOpener
from .processor import NoOpCubeProcessor
from .usercode import CubeUserCodeExecutor
from .writer import CubeWriter
from ..generator import CubeGenerator
from ..progress import ApiProgressCallbackObserver
from ..progress import ConsoleProgressObserver
from ..request import CubeGeneratorRequest
from ..request import CubeGeneratorRequestLike
from ..response import CubeInfo


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
            code_executor = CubeUserCodeExecutor(request.code_config)
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
