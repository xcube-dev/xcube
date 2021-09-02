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
from .helpers import is_empty_cube
from .informant import CubeInformant
from .opener import DatasetsOpener
from .rechunker import CubeRechunker
from .resamplert import CubeResamplerT
from .resamplerxy import CubeResamplerXY
from .subsetter import CubeSubsetter
from .transformer import CubeIdentity
from .transformer import transform_cube
from .usercode import CubeUserCodeExecutor
from .writer import CubeWriter
from ..generator import CubeGenerator
from ..progress import ApiProgressCallbackObserver
from ..progress import ConsoleProgressObserver
from ..request import CubeGeneratorRequest
from ..request import CubeGeneratorRequestLike
from ..response import CubeGeneratorResult
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
        self._generated_data_id: Optional[str] = None
        self._generated_cube: Optional[xr.Dataset] = None

    @property
    def generated_data_id(self) -> Optional[str]:
        return self._generated_data_id

    @property
    def generated_cube(self) -> Optional[xr.Dataset]:
        return self._generated_cube

    def generate_cube(self, request: CubeGeneratorRequestLike) \
            -> CubeGeneratorResult:
        request = CubeGeneratorRequest.normalize(request)
        request = request.for_local()

        # noinspection PyUnusedLocal
        def _no_op_callable(ds, **kwargs):
            return ds

        if request.callback_config:
            ApiProgressCallbackObserver(request.callback_config).activate()

        if self._verbosity:
            ConsoleProgressObserver().activate()

        cubes_opener = DatasetsOpener(request.cube_config,
                                      store_pool=self._store_pool)

        cube_subsetter = CubeSubsetter()
        cube_resampler_xy = CubeResamplerXY()
        cube_resampler_t = CubeResamplerT()
        cube_combiner = CubesCombiner(request.cube_config)
        cube_rechunker = CubeRechunker()

        if request.code_config is not None:
            code_executor = CubeUserCodeExecutor(request.code_config)
        else:
            code_executor = CubeIdentity()

        cube_writer = CubeWriter(request.output_config,
                                 store_pool=self._store_pool)

        num_inputs = len(request.input_configs)
        # Estimated workload:
        opener_work = 5
        resampler_t_work = 10
        resampler_xy_work = 20
        subsetter_work = 1
        combiner_work = 1
        rechunker_work = 1
        executor_work = 1
        writer_work = 80
        total_work = (opener_work
                      + subsetter_work
                      + resampler_t_work
                      + resampler_xy_work) * num_inputs \
                     + rechunker_work \
                     + executor_work \
                     + writer_work

        t_cubes = []
        with observe_progress('Processing input dataset(s)',
                              total_work) as progress:
            for input_config in request.input_configs:
                progress.will_work(opener_work)
                t_cube = cubes_opener.open_cube(input_config)

                progress.will_work(subsetter_work)
                t_cube = transform_cube(t_cube, cube_subsetter)

                progress.will_work(resampler_t_work)
                t_cube = transform_cube(t_cube, cube_resampler_t)

                progress.will_work(resampler_xy_work)
                t_cube = transform_cube(t_cube, cube_resampler_xy)

                t_cubes.append(t_cube)

            progress.will_work(combiner_work)
            t_cube = cube_combiner.combine_cubes(t_cubes)

            progress.will_work(rechunker_work)
            t_cube = transform_cube(t_cube, cube_rechunker)

            progress.will_work(executor_work)
            t_cube = transform_cube(t_cube, code_executor)

            progress.will_work(rechunker_work)
            t_cube = transform_cube(t_cube, cube_rechunker)

            progress.will_work(writer_work)
            cube, gm, _ = t_cube
            if not is_empty_cube(cube):
                data_id, cube = cube_writer.write_cube(cube, gm)
                self._generated_data_id = data_id
                self._generated_cube = cube
                status = 'ok'
                message = None
            else:
                data_id = request.output_config.data_id
                status = 'warning'
                message = 'An empty cube has been generated,' \
                          ' hence it has not been written at all.'

        if self._verbosity:
            print('Cube "{}" generated within {:.2f} seconds'
                  .format(str(data_id), progress.state.total_time))

        return CubeGeneratorResult(data_id=data_id,
                                   status=status,
                                   message=message)

    def get_cube_info(self, request: CubeGeneratorRequestLike) -> CubeInfo:
        request = CubeGeneratorRequest.normalize(request)
        informant = CubeInformant(request=request.for_local(),
                                  store_pool=self._store_pool)
        return informant.generate()
