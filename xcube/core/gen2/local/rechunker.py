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
from typing import Tuple, Optional

import xarray as xr

from xcube.core.gridmapping import GridMapping
from xcube.core.schema import get_dataset_chunks
from .transformer import CubeTransformer
from .transformer import TransformedCube
from ..config import CubeConfig


class CubeRechunker(CubeTransformer):
    """Force cube to have chunks compatible with Zarr."""

    def transform_cube(self,
                       cube: xr.Dataset,
                       gm: GridMapping,
                       cube_config: CubeConfig) -> TransformedCube:

        # get initial cube chunks from existing dataset
        cube_chunks = get_dataset_chunks(cube)

        # Get potential tile size
        tile_size: Optional[Tuple[int, int]] = None
        if cube_config.tile_size is not None:
            tile_size = cube_config.tile_size
        elif gm.tile_size is not None:
            tile_size = gm.tile_size

        # if tile sizes are specified, use them to overwrite
        # the spatial dimensions
        x_dim_name, y_dim_name = gm.xy_dim_names
        if tile_size is not None:
            tile_width, tile_height = tile_size
            for dim_name, size in ((x_dim_name, tile_width),
                                   (y_dim_name, tile_height)):
                if dim_name not in cube_chunks:
                    cube_chunks[dim_name] = size

        # cube_config.chunks will overwrite any defaults
        if cube_config.chunks:
            for dim_name, chunks in cube_config.chunks:
                cube_chunks[dim_name] = chunks

        # If there is no chunking, return identities
        if not cube_chunks:
            return cube, gm, cube_config

        chunked_cube = xr.Dataset(attrs=cube.attrs)

        # Coordinate variables are chunked automatically
        chunked_cube = chunked_cube.assign_coords(
            coords={
                var_name: var.chunk({
                    dim_name: 'auto'
                    for dim_name in var.dims
                })
                for var_name, var in cube.coords.items()
            }
        )

        # Data variables are chunked according to cube_chunks
        chunked_cube = chunked_cube.assign(
            variables={
                var_name: var.chunk({
                    dim_name: cube_chunks.get(dim_name, 'auto')
                    for dim_name in var.dims
                })
                for var_name, var in cube.data_vars.items()
            }
        )

        # Test whether tile size has changed after re-chunking.
        # If so, we must change the grid mapping too.
        tile_width = cube_chunks.get(x_dim_name)
        tile_height = cube_chunks.get(y_dim_name)
        assert tile_width is not None
        assert tile_height is not None
        tile_size = (tile_width, tile_height)
        if tile_size != gm.tile_size:
            # Note, changing grid mapping tile size may
            # rechunk (2D) coordinates in chunked_cube too
            gm = gm.derive(tile_size=tile_size)

        # Update chunks encoding for Zarr
        for var in chunked_cube.variables.values():
            if var.chunks is not None:
                # sizes[0] is the first of
                # e.g. sizes = (512, 512, 71)
                var.encoding.update(chunks=[
                    sizes[0] for sizes in var.chunks
                ])
            elif 'chunks' in var.encoding:
                del var.encoding['chunks']

        return chunked_cube, gm, cube_config
