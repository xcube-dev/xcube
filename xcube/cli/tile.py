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

from typing import List, Optional, Mapping, Any

import click

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

DEFAULT_OUTPUT_PATH = 'out.tiles'
DEFAULT_CONFIG_PATH = 'config.yml'
DEFAULT_STYLE_ID = 'default'


@click.command(name='tile')
@click.argument('cube', nargs=1)
@click.option('--variables', '--vars', metavar='VARIABLES',
              help='Variables to be included in output. '
                   'Comma-separated list of names which may contain wildcard characters "*" and "?".')
@click.option('--labels', 'labels', metavar='LABELS',
              help=f'Labels for non-spatial dimensions, e.g. "time=2019-20-03".'
                   f' Multiple values are separated by comma.')
@click.option('--tile-size', '-t', 'tile_size', metavar='TILE_SIZE',
              help=f'Tile size in pixels for individual or both x- and y-directions. '
                   f'Separate by comma for individual tile sizes, e.g. "-t 360,180". '
                   f'Defaults to the chunks sizes in x- and y-directions of CUBE, '
                   f'which may not be ideal. Use option --dry-run and --verbose to '
                   f'display the default tile sizes for CUBE.')
@click.option('--config', '-c', 'config_path', metavar='CONFIG',
              help=f'Configuration file in YAML format.')
@click.option('--style', '-s', 'style_id', metavar='STYLE', default=DEFAULT_STYLE_ID,
              help=f'Name of a style identifier in CONFIG file. Only used if CONFIG is given.'
                   f' Defaults to {DEFAULT_STYLE_ID!r}.')
@click.option('--output', '-o', 'output_path', metavar='OUTPUT', default=DEFAULT_OUTPUT_PATH,
              help=f'Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}')
@click.option('--verbose', '-v', is_flag=True, multiple=True,
              help=f'Use -vv to report all files generated, -v to report less.')
@click.option('--dry-run', 'dry_run', is_flag=True,
              help=f'Generate all tiles but don\'t write any files.')
def tile(cube: str,
         variables: Optional[str],
         labels: Optional[str],
         tile_size: Optional[str],
         config_path: Optional[str],
         style_id: Optional[str],
         output_path: Optional[str],
         verbose: List[bool],
         dry_run: bool):
    """
    Create RGBA tiles from CUBE.

    Color bars and value ranges for variables can be specified in a CONFIG file.
    Here the color mappings are defined for a style named "ocean_color":

    \b
    Styles:
      - Identifier: ocean_color
        ColorMappings:
          conc_chl:
            ColorBar: "plasma"
            ValueRange: [0., 24.]
          conc_tsm:
            ColorBar: "PuBuGn"
            ValueRange: [0., 100.]
          kd489:
            ColorBar: "jet"
            ValueRange: [0., 6.]

    This is the same styles syntax as the configuration file for "xcube serve",
    hence its configuration can be reused.

    """
    import fractions
    import itertools
    import json
    import os.path
    # noinspection PyPackageRequirements
    import yaml
    import xarray as xr
    import numpy as np

    from xcube.core.mldataset import open_ml_dataset
    from xcube.core.schema import CubeSchema
    from xcube.core.tile import get_ml_dataset_tile
    from xcube.core.tile import parse_non_spatial_labels
    from xcube.core.select import select_variables_subset
    from xcube.cli.common import parse_cli_kwargs
    from xcube.cli.common import parse_cli_sequence
    from xcube.cli.common import assert_positive_int_item
    from xcube.util.tilegrid import TileGrid
    from xcube.util.tiledimage import DEFAULT_COLOR_MAP_NAME
    from xcube.util.tiledimage import DEFAULT_COLOR_MAP_VALUE_RANGE
    from xcube.util.tiledimage import DEFAULT_COLOR_MAP_NUM_COLORS

    # noinspection PyShadowingNames
    def write_tile_map_resource(path: str,
                                resolutions: List[fractions.Fraction],
                                tile_grid: TileGrid,
                                title='',
                                abstract='',
                                srs='CRS:84'):
        num_levels = len(resolutions)
        z_and_upp = zip(range(num_levels), map(float, resolutions))
        x1, y1, x2, y2 = tile_grid.geo_extent
        xml = [f'<TileMap version="1.0.0" tilemapservice="http://tms.osgeo.org/1.0.0">',
               f'  <Title>{title}</Title>',
               f'  <Abstract>{abstract}</Abstract>',
               f'  <SRS>{srs}</SRS>',
               f'  <BoundingBox minx="{x1}" miny="{y1}" maxx="{x2}" maxy="{y2}"/>',
               f'  <Origin x="{x1}" y="{y1}"/>',
               f'  <TileFormat width="{tile_grid.tile_width}" height="{tile_grid.tile_height}"'
               f' mime-type="image/png" extension="png"/>',
               f'  <TileSets profile="local">'] + [
                  f'    <TileSet href="{z}" order="{z}" units-per-pixel="{upp}"/>' for z, upp in z_and_upp] + [
                  f'  </TileSets>',
                  f'</TileMap>']
        with open(path, 'w') as fp:
            fp.write('\n'.join(xml))

    # noinspection PyShadowingNames
    def _convert_coord_var(coord_var: xr.DataArray):
        values = coord_var.values
        if np.issubdtype(values.dtype, np.datetime64):
            return list(np.datetime_as_string(values, timezone='UTC'))
        elif np.issubdtype(values.dtype, np.integer):
            return [int(value) for value in values]
        else:
            return [float(value) for value in values]

    # noinspection PyShadowingNames
    def _get_color_mappings(var_name: str, config: Mapping[str, Any], style_id: str):
        color_bar = DEFAULT_COLOR_MAP_NAME
        value_range = DEFAULT_COLOR_MAP_VALUE_RANGE
        if config:
            style_id = style_id or 'default'
            styles = config.get('Styles')
            if styles:
                color_mappings = None
                for style in styles:
                    if style.get('Identifier') == style_id:
                        color_mappings = style.get('ColorMappings')
                        break
                if color_mappings:
                    color_mapping = color_mappings.get(var_name)
                    if color_mapping:
                        color_bar = color_mapping.get('ColorBar', color_bar)
                        value_range = color_mapping.get('ValueRange', value_range)
        value_min, value_max = value_range
        return color_bar, value_min, value_max

    variables = parse_cli_sequence(variables, metavar='VARIABLES', num_items_min=1,
                                   item_plural_name='variables')

    tile_size = parse_cli_sequence(tile_size, num_items=2, metavar='TILE_SIZE',
                                   item_parser=int,
                                   item_validator=assert_positive_int_item,
                                   item_plural_name='tile sizes')

    labels = parse_cli_kwargs(labels, metavar='LABELS')

    verbosity = len(verbose)

    config = {}
    if config_path:
        if verbosity:
            print(f'Opening {config_path}...')
        with open(config_path, 'r') as fp:
            config = yaml.safe_load(fp)

    if verbosity:
        print(f'Opening {cube}...')

    ml_dataset = open_ml_dataset(cube, chunks='auto')
    tile_grid = ml_dataset.tile_grid
    base_dataset = ml_dataset.base_dataset
    schema = CubeSchema.new(base_dataset)
    spatial_dims = schema.x_dim, schema.y_dim

    if tile_size:
        tile_width, tile_height = tile_size
    else:
        if verbosity:
            print(f'Warning: using default tile sizes derived from CUBE')
        tile_width, tile_height = tile_grid.tile_width, tile_grid.tile_height

    indexers = None
    if labels:
        indexers = parse_non_spatial_labels(labels,
                                            schema.dims,
                                            schema.coords,
                                            allow_slices=True,
                                            exception_type=click.ClickException)

    def transform(ds: xr.Dataset) -> xr.Dataset:
        if variables:
            ds = select_variables_subset(ds, var_names=variables)
        if indexers:
            ds = ds.sel(**indexers)
        chunk_sizes = {dim: 1 for dim in ds.dims}
        chunk_sizes[spatial_dims[0]] = tile_width
        chunk_sizes[spatial_dims[1]] = tile_height
        return ds.chunk(chunk_sizes)

    ml_dataset = ml_dataset.apply(transform)
    tile_grid = ml_dataset.tile_grid
    base_dataset = ml_dataset.base_dataset
    schema = CubeSchema.new(base_dataset)
    spatial_dims = schema.x_dim, schema.y_dim

    x1, _, x2, _ = tile_grid.geo_extent
    num_levels = tile_grid.num_levels
    resolutions = [fractions.Fraction(fractions.Fraction(x2 - x1), tile_grid.width(z))
                   for z in range(num_levels)]

    if verbosity:
        print(f'Writing tile sets...')
        print(f'  Zoom levels: {num_levels}')
        print(f'  Resolutions: {", ".join(map(str, resolutions))} units/pixel')
        print(f'  Tile size:   {tile_width} x {tile_height} pixels')

    image_cache = {}

    for var_name, var in base_dataset.data_vars.items():
        color_bar, value_min, value_max = _get_color_mappings(str(var_name), config, style_id)

        label_names = []
        label_indexes = []
        for dim in var.dims:
            if dim not in spatial_dims:
                label_names.append(dim)
                label_indexes.append(list(range(var[dim].size)))

        var_path = os.path.join(output_path, str(var_name))
        metadata_path = os.path.join(var_path, 'metadata.json')
        metadata = dict(name=str(var_name),
                        attrs={name: value
                               for name, value in var.attrs.items()},
                        dims=[str(dim)
                              for dim in var.dims],
                        dim_sizes={dim: int(var[dim].size)
                                   for dim in var.dims},
                        color_mapping=dict(color_bar=color_bar,
                                           value_min=value_min,
                                           value_max=value_max,
                                           num_colors=DEFAULT_COLOR_MAP_NUM_COLORS),
                        coordinates={name: _convert_coord_var(coord_var)
                                     for name, coord_var in var.coords.items() if coord_var.ndim == 1})
        if verbosity:
            print(f'Writing {metadata_path}')
        if not dry_run:
            os.makedirs(var_path, exist_ok=True)
            with open(metadata_path, 'w') as fp:
                json.dump(metadata, fp, indent=2)

        for label_index in itertools.product(*label_indexes):
            labels = {name: index for name, index in zip(label_names, label_index)}
            tilemap_path = os.path.join(var_path, *[str(l) for l in label_index])
            tilemap_resource_path = os.path.join(tilemap_path, 'tilemapresource.xml')
            if verbosity > 1:
                print(f'Writing {tilemap_resource_path}')
            if not dry_run:
                os.makedirs(tilemap_path, exist_ok=True)
                write_tile_map_resource(tilemap_resource_path, resolutions, tile_grid, title=f'{var_name}')
            for z in range(num_levels):
                num_tiles_x = tile_grid.num_tiles_x(z)
                num_tiles_y = tile_grid.num_tiles_y(z)
                tile_z_path = os.path.join(tilemap_path, str(z))
                if not dry_run and not os.path.exists(tile_z_path):
                    os.mkdir(tile_z_path)
                for x in range(num_tiles_x):
                    tile_zx_path = os.path.join(tile_z_path, str(x))
                    if not dry_run and not os.path.exists(tile_zx_path):
                        os.mkdir(tile_zx_path)
                    for y in range(num_tiles_y):
                        tile_bytes = get_ml_dataset_tile(ml_dataset,
                                                         str(var_name),
                                                         x, y, z,
                                                         labels=labels,
                                                         labels_are_indices=True,
                                                         cmap_cbar=color_bar,
                                                         cmap_vmin=value_min,
                                                         cmap_vmax=value_max,
                                                         image_cache=image_cache,
                                                         trace_perf=True,
                                                         exception_type=click.ClickException)
                        tile_path = os.path.join(tile_zx_path, f'{num_tiles_y - 1 - y}.png')
                        if verbosity > 2:
                            print(f'Writing tile {tile_path}')
                        if not dry_run:
                            with open(tile_path, 'wb') as fp:
                                fp.write(tile_bytes)

    print(f'Done writing tile sets.')
