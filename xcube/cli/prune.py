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

import os
import os.path

import click

from xcube.constants import FORMAT_NAME_ZARR


# noinspection PyShadowingBuiltins
@click.command(name='prune')
@click.argument('cube')
@click.option('--dry-run', is_flag=True,
              help='Just read and process input, but don\'t produce any outputs.')
def prune(cube, dry_run):
    """
    Delete empty chunks.
    Deletes all data files associated with empty (NaN-only) chunks in given CUBE,
    which must have ZARR format.
    """
    _prune(input_path=cube, dry_run=dry_run, monitor=print)
    return 0


def _prune(input_path: str = None,
           dry_run: bool = False,
           monitor=None):
    from xcube.core.chunk import get_empty_dataset_chunks
    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_cube

    input_format = guess_dataset_format(input_path)
    if input_format != FORMAT_NAME_ZARR:
        raise click.ClickException("input must be a cube in ZARR format")

    monitor(f'Opening cube from {input_path!r}...')
    with open_cube(input_path) as cube:
        monitor('Identifying empty blocks...')
        empty_chunks = get_empty_dataset_chunks(cube)

    num_deleted = 0
    for var_name, chunk_indices in empty_chunks.items():
        monitor(f'Deleting {len(chunk_indices)} empty block file(s) for variable {var_name!r}...')
        for chunk_index in chunk_indices:
            ok = _delete_block_file(input_path, var_name, chunk_index, dry_run, monitor)
            if ok:
                num_deleted += 1

    monitor(f'Done, {num_deleted} block file(s) deleted.')


def _delete_block_file(input_path, var_name, chunk_index, dry_run, monitor) -> bool:
    block_path = None
    block_path_1 = block_path_2 = os.path.join(input_path, var_name, '.'.join(map(str, chunk_index)))
    if os.path.isfile(block_path_1):
        block_path = block_path_1
    else:
        block_path_2 = os.path.join(input_path, var_name, *map(str, chunk_index))
        if os.path.isfile(block_path_2):
            block_path = block_path_2
    if block_path:
        if dry_run:
            return True
        try:
            os.remove(block_path)
            return True
        except OSError as e:
            monitor(f'error: failed to delete block file {block_path}: {e}')
    else:
        monitor(f'error: could neither find block file {block_path_1} nor {block_path_2}')
    return False
