# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

from typing import Tuple, Dict, List

import click
import os.path
import os
# TODO (forman): move FORMAT_NAME_ZARR to constants,
from xcube.util.dsio import FORMAT_NAME_ZARR


# noinspection PyShadowingBuiltins
@click.command(name='prune')
@click.argument('input')
@click.option('--dry-run', default=False, is_flag=True,
              help='Just read and process inputs, but don\'t produce any outputs.')
def prune(input, dry_run):
    """
    Delete empty chunks.
    """
    _prune(input_path=input, dry_run=dry_run, monitor=print)
    return 0


def _prune(input_path: str = None,
           dry_run: bool = False,
           monitor=None):
    from xcube.api import open_cube
    from xcube.api.readwrite import write_cube
    from xcube.util.dsio import guess_dataset_format
    import xarray as xr
    import numpy as np
    import dask.array

    # TODO (forman): make this API
    def get_empty_chunks(cube: xr.Dataset) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Identify empty cube chunks and return their indices.

        :param cube: The cube.
        :return: A mapping from variable name to a list of chunk indices.
        """
        for var_name in cube.data_vars:
            var = cube[var_name]

            xr.DataArray(dask.array.fromfunction(), dims=var.dims, name=)

            def apply_function(data_block, index_block):
                if np.all(np.isnan(data_block)):

            xr.apply_ufunc(apply_function, var)

    input_format = guess_dataset_format(input_path)
    if input_format != FORMAT_NAME_ZARR:
        raise click.ClickException("input must be a cube in ZARR format")

    monitor(f'Opening cube from {input_path!r}...')
    with open_cube(input_path) as cube:

        monitor('Identifying empty chunks...')
        empty_chunks = get_empty_chunks(cube)

        num_empty_chunks = 0
        for var_name, chunk_indices in empty_chunks:
            num_empty_chunks += len(chunk_indices)

        for var_name, chunk_indices in empty_chunks:
            monitor(f'Deleting {len(chunk_indices)} empty chunk file(s) for variable {var_name!r}...')
            if not dry_run:
                for chunk_index in chunk_indices:
                    chunk_path = os.path.join(input_path, var_name, ".".join(chunk_index))
                    os.remove(chunk_path)

        monitor(f'Done.')



