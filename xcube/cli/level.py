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

import click

DEFAULT_TILE_SIZE = 512


# noinspection PyShadowingBuiltins
@click.command(name="level")
@click.argument('input')
@click.option('--output', '-o', metavar='OUTPUT',
              help='Output path. If omitted, "INPUT.levels" will be used.')
@click.option('--link', '-L', is_flag=True, flag_value=True,
              help='Link the INPUT instead of converting it to a level'
                   ' zero dataset. Use with care, as the INPUT\'s internal'
                   ' spatial chunk sizes may be inappropriate'
                   ' for imaging purposes.')
@click.option('--tile-size', '-t', metavar='TILE_SIZE',
              help=f'Tile size, given as single integer number or'
                   f' as <tile-width>,<tile-height>. '
                   f'If omitted, the tile size will be derived'
                   f' from the INPUT\'s'
                   f' internal spatial chunk sizes.'
                   f' If the INPUT is not chunked,'
                   f' tile size will be {DEFAULT_TILE_SIZE}.')
@click.option('--num-levels-max', '-n', metavar='NUM_LEVELS_MAX', type=int,
              help=f'Maximum number of levels to generate. '
                   f'If not given, the number of levels will'
                   f' be derived from spatial dimension and tile sizes.')
def level(input, output, link, tile_size, num_levels_max):
    """
    Generate multi-resolution levels.
    Transform the given dataset by INPUT into the levels of a
    multi-level pyramid with spatial resolution decreasing by a
    factor of two in both spatial dimensions and write the
    result to directory OUTPUT.
    """
    import time
    import os
    from xcube.cli.common import parse_cli_sequence
    from xcube.cli.common import assert_positive_int_item
    from xcube.core.level import write_levels

    input_path = input
    output_path = output
    link_input = link

    if num_levels_max is not None and num_levels_max < 1:
        raise click.ClickException(
            f"NUM_LEVELS_MAX must be a positive integer"
        )

    if not output_path:
        dir_path = os.path.dirname(input_path)
        basename, ext = os.path.splitext(os.path.basename(input_path))
        output_path = os.path.join(dir_path, basename + ".levels")

    if os.path.exists(output_path):
        raise click.ClickException(
            f"output {output_path!r} already exists"
        )

    spatial_tile_shape = None
    if tile_size is not None:
        tile_size = parse_cli_sequence(
            tile_size,
            metavar='TILE_SIZE', num_items=2,
            item_plural_name='tile sizes',
            item_parser=int,
            item_validator=assert_positive_int_item
        )
        spatial_tile_shape = tile_size[1], tile_size[0]

    start_time = t0 = time.perf_counter()

    # noinspection PyUnusedLocal
    def progress_monitor(dataset, index, num_levels):
        nonlocal t0
        print(f"Level {index + 1} of {num_levels} written"
              f" after {time.perf_counter() - t0} seconds")
        t0 = time.perf_counter()

    levels = write_levels(output_path,
                          input_path=input_path,
                          link_input=link_input,
                          progress_monitor=progress_monitor,
                          spatial_tile_shape=spatial_tile_shape,
                          num_levels_max=num_levels_max)
    print(f"{len(levels)} level(s) written into {output_path}"
          f" after {time.perf_counter() - start_time} seconds")
