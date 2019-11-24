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
from typing import List

import click

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

DEFAULT_OUTPUT_PATH = 'out.tiles'

DEFAULT_COLOR_BAR_NAME = 'viridis'
DEFAULT_COLOR_BAR_VMIN = 0.
DEFAULT_COLOR_BAR_VMAX = 1.


@click.command(name='tile')
@click.argument('cube', nargs=1)
@click.option('--variables', '--vars', metavar='VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--labels', 'raw_labels', metavar='LABELS',
              help=f'Labels for non-spatial dimensions, e.g. "time=2019-20-03".'
                   f' Multiple values are separated by comma.')
@click.option('--cbar', 'color_bar_name', metavar='COLOR_BAR', default=DEFAULT_COLOR_BAR_NAME,
              help=f'Color bar name. Must be the name of a valid matplotlib color bar.'
                   f' Defaults to {DEFAULT_COLOR_BAR_NAME}.')
@click.option('--vmin', 'color_bar_vmin', metavar='MIN_VALUE', default=DEFAULT_COLOR_BAR_VMIN, type=float,
              help=f'Variable value that maps to first color of the color bar.'
                   f' Defaults to {DEFAULT_COLOR_BAR_VMIN}.')
@click.option('--vmax', 'color_bar_vmax', metavar='MAX_VALUE', default=DEFAULT_COLOR_BAR_VMAX, type=float,
              help=f'Variable value that maps to last color of the color bar.'
                   f' Defaults to {DEFAULT_COLOR_BAR_VMAX}.')
@click.option('--output', '-o', 'output_path', metavar='OUTPUT', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--dry-run', 'dry_run', is_flag=True,
              help=f'Generate all tiles but don\'t write any files.')
def tile(cube: str,
         variables: str,
         raw_labels: str,
         color_bar_name: str,
         color_bar_vmin: float,
         color_bar_vmax: float,
         output_path: str,
         dry_run: bool):
    """
    Create RGBA tiles from CUBE.
    """
    import itertools
    import os.path

    import numpy as np

    from xcube.core.extract import get_dataset_indexes
    from xcube.core.mldataset import open_ml_dataset
    from xcube.core.schema import CubeSchema
    from xcube.core.tile import get_ml_dataset_tile
    from xcube.core.tile import parse_non_spatial_labels
    from xcube.cli.common import parse_cli_kwargs
    from xcube.util.tilegrid import TileGrid

    def write_tile_map_resource(path: str,
                                resolutions: List[float],
                                tile_grid: TileGrid,
                                title='',
                                abstract='',
                                srs='EPSG:4326'):
        num_levels = len(resolutions)
        z_and_upp = zip(range(num_levels), resolutions)
        if srs == 'EPSG:4326':
            y1, x1, y2, x2 = tile_grid.geo_extent
        else:
            x1, y1, x2, y2 = tile_grid.geo_extent
        xml = [f'<TileMap version="1.0.0" tilemapservice="http://tms.osgeo.org/1.0.0">',
               f'  <Title>{title}</Title>',
               f'  <Abstract>{abstract}</Abstract>',
               f'  <SRS>{srs}</SRS>',
               f'  <BoundingBox minx="{x1}" miny="{y1}" maxx="{x2}" maxy="{y2}"/>',
               f'  <Origin x="{x1}" y="{y1}"/>',
               f'  <TileFormat width="{tile_grid.tile_width}" height="{tile_grid.tile_height}"'
               f' mime-type="image/png" extension="png"/>',
               f'  <TileSets profile="geodetic">\n'] + [
                  f'    <TileSet href="{z}" order="{z}" units-per-pixel="{upp}"/>\n' for z, upp in z_and_upp] + [
                  f'  </TileSets>',
                  f'</TileMap>\n']
        with open(path, 'w') as fp:
            fp.write('\n'.join(xml))


    variables = variables or None
    if variables is not None:
        try:
            variables = list(map(lambda c: str(c).strip(), variables.split(',')))
        except ValueError:
            variables = None
        if variables is not None \
                and next(iter(True for var_name in variables if var_name == ''), False):
            variables = None
        if variables is None or len(variables) == 0:
            raise click.ClickException(f'invalid variables {variables!r}')

    raw_labels = parse_cli_kwargs(raw_labels, 'LABELS')

    print(f'opening {cube}...')
    ml_dataset = open_ml_dataset(cube)

    dataset = ml_dataset.base_dataset
    variables = set(variables or dataset.data_vars)
    schema = CubeSchema.new(dataset)
    dims = list(schema.dims)
    dims.remove(schema.x_dim)
    dims.remove(schema.y_dim)
    tile_grid = ml_dataset.tile_grid

    image_cache = {}

    for var_name, var in dataset.data_vars.items():
        if var_name not in variables:
            continue

        labels = {}
        if raw_labels:
            default_labels = parse_non_spatial_labels(var,
                                                      raw_labels,
                                                      exception_type=click.ClickException)
            for name, value in default_labels.items():
                indexes = get_dataset_indexes(dataset, name, np.array([value]))
                index = int(indexes[0])
                if index < 0:
                    raise ValueError(f'label value {value!r} for dimension {index!r} is out of range')
                labels[name] = index

        label_names = []
        label_indexes = []
        for dim in var.dims:
            if dim not in (schema.x_dim, schema.y_dim):
                label_names.append(dim)
                if dim in labels:
                    label_indexes.append([labels[dim]])
                else:
                    label_indexes.append(list(range(var[dim].size)))

        var_path = os.path.join(output_path, str(var_name))
        metadata_path = os.path.join(var_path, 'metadata.json')
        print(f'writing {metadata_path}')
        if not dry_run:
            os.makedirs(var_path, exist_ok=True)
            # TODO: write metadata_path incl. coordinates, dims
            pass

        for label_index in itertools.product(*label_indexes):
            labels = {name: index for name, index in zip(label_names, label_index)}
            tilemap_path = os.path.join(var_path, *[str(l) for l in label_index])
            tilemap_resource_path = os.path.join(tilemap_path, 'tilemapresource.xml')
            print(f'writing {tilemap_resource_path}')
            if not dry_run:
                os.makedirs(tilemap_path, exist_ok=True)
                # TODO: compute correct resolutions
                write_tile_map_resource(tilemap_resource_path, [0.1, 0.2, 0.4], tile_grid, title=f'{var_name}')
            for level in range(tile_grid.num_levels):
                for y in range(tile_grid.num_tiles_y(level)):
                    for x in range(tile_grid.num_tiles_x(level)):
                        z = tile_grid.num_levels - 1 - level
                        tile = get_ml_dataset_tile(ml_dataset,
                                                   str(var_name),
                                                   x, y, z,
                                                   labels=labels,
                                                   labels_are_indices=True,
                                                   cmap_cbar=color_bar_name,
                                                   cmap_vmin=color_bar_vmin,
                                                   cmap_vmax=color_bar_vmax,
                                                   image_cache=image_cache,
                                                   trace_perf=True,
                                                   exception_type=click.ClickException)
                        tile_zy_path = os.path.join(tilemap_path, *[str(i) for i in (z, y)])
                        tile_path = os.path.join(tile_zy_path, f'{x}.png')
                        print(f'writing tile {tile_path}')
                        if not dry_run:
                            os.makedirs(tile_zy_path, exist_ok=True)
                            with open(tile_path, 'wb') as fp:
                                fp.write(tile)
