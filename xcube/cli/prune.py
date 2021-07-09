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
from typing import Tuple, Callable

import click

from xcube.constants import FORMAT_NAME_ZARR

Monitor = Callable[[str, int], None]


# noinspection PyShadowingBuiltins
@click.command(name='prune')
@click.argument('dataset_path', metavar='DATASET')
@click.option('--verbose', '-v', is_flag=True, multiple=True,
              help='Verbose mode. Multiple may be given, '
                   'for example "-vvv".')
@click.option('--dry-run', is_flag=True,
              help='Just read and process input, '
                   'but don\'t produce any output.')
def prune(dataset_path: str,
          verbose: Tuple[bool, ...],
          dry_run: bool):
    """
    Delete empty chunks.
    Deletes all data files associated with empty
    (NaN-only) chunks in given DATASET,
    which must have Zarr format.
    """
    max_verbosity = sum(verbose)

    def monitor(msg: str, verbosity: int = 1):
        if verbosity <= max_verbosity:
            print(msg)

    _prune(input_path=dataset_path, dry_run=dry_run, monitor=monitor)
    return 0


def _prune(input_path: str,
           dry_run: bool,
           monitor: Monitor):
    from xcube.core.chunk import get_empty_dataset_chunks
    from xcube.core.dsio import guess_dataset_format
    from xcube.core.dsio import open_dataset

    input_format = guess_dataset_format(input_path)
    if input_format != FORMAT_NAME_ZARR:
        raise click.ClickException("input must be a dataset in Zarr format")

    num_deleted_total = 0

    monitor(f'Opening dataset from {input_path!r}...', 1)
    with open_dataset(input_path) as dataset:
        monitor('Identifying empty chunks...', 1)
        for var_name, chunk_indices in get_empty_dataset_chunks(dataset):
            num_empty_chunks = 0
            num_deleted = 0
            for chunk_index in chunk_indices:
                num_empty_chunks += 1
                if num_empty_chunks == 1:
                    monitor(f'Found empty chunks in variable {var_name!r}, '
                            f'deleting block files...', 2)

                ok = _delete_block_file(input_path, var_name, chunk_index,
                                        dry_run, monitor)
                if ok:
                    num_deleted += 1
            if num_deleted > 0:
                monitor(f'Deleted {num_deleted} block file(s) '
                        f'for variable {var_name!r}.', 2)
            elif num_empty_chunks > 0:
                monitor(f'No block files for variable {var_name!r} '
                        f'could be deleted.', 2)
            num_deleted_total += num_deleted

    monitor(f'Done, {num_deleted_total} block file(s) deleted total.', 1)


def _delete_block_file(input_path: str,
                       var_name: str,
                       chunk_index,
                       dry_run: bool,
                       monitor: Monitor) -> bool:
    block_path = None
    block_path_1 = os.path.join(input_path,
                                var_name,
                                '.'.join(map(str, chunk_index)))
    block_path_2 = block_path_1

    if os.path.isfile(block_path_1):
        block_path = block_path_1
    else:
        block_path_2 = os.path.join(input_path,
                                    var_name,
                                    *map(str, chunk_index))
        if os.path.isfile(block_path_2):
            block_path = block_path_2

    if block_path:
        monitor(f'Deleting chunk file {block_path}', 3)
        if dry_run:
            return True
        try:
            os.remove(block_path)
            return True
        except OSError as e:
            monitor(f'Error: failed to delete '
                    f'block file {block_path}: {e}', 0)
    else:
        monitor(f'Error: could find neither '
                f'block file {block_path_1} nor {block_path_2}', 0)
    return False
