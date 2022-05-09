# The MIT License (MIT)
# Copyright (c) 2021/2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional, List, Union, Tuple

import rasterio
import rioxarray
import xarray as xr

from .gridmapping import GridMapping
from .mrdataset import MultiResDataset


class GeoTIFFMultiResDataset(MultiResDataset):

    @classmethod
    def open_dataset(cls,
                     file_spec: str,
                     tile_size: Tuple[int, int],
                     overview_level: Optional[int] = 0) -> xr.Dataset:
        array: xr.DataArray = rioxarray.open_rasterio(
            file_spec,
            overview_level=overview_level,
            chunks=dict(zip(('x', 'y'), tile_size))
        )
        arrays = {}
        if array.ndim == 3:
            for i in range(array.shape[0]):
                name = f'{array.name or "band"}_{i + 1}'
                dims = array.dims[-2:]
                coords = {n: v
                          for n, v in array.coords.items()
                          if n in dims or n == 'spatial_ref'}
                band_data = array.data[i, :, :]
                arrays[name] = xr.DataArray(band_data,
                                            coords=coords,
                                            dims=dims,
                                            attrs=dict(**array.attrs))
        elif array.ndim == 2:
            name = f'{array.name or "band"}'
            arrays[name] = array
        else:
            raise RuntimeError('number of dimensions must be 2 or 3')

        dataset = xr.Dataset(arrays, attrs=dict(source=file_spec))
        # For CRS, rioxarray uses variable "spatial_ref" by default
        if 'spatial_ref' in array.coords:
            for data_var in dataset.data_vars.values():
                data_var.attrs['grid_mapping'] = 'spatial_ref'

        return dataset

    def __init__(self,
                 path_or_url: str,
                 tile_size: Optional[Union[int, Tuple[int, int]]] = None):

        with rasterio.open(path_or_url) as rio_dataset:
            overviews = rio_dataset.overviews(1)

        num_levels = len(overviews) + 1

        dataset = self.open_dataset(path_or_url, tile_size or (512, 512))

        grid_mapping = GridMapping.from_dataset(dataset)

        # determine sizes and num_levels from
        # tile_size and coordinate array sizes
        x_res, y_res = grid_mapping.res
        resolutions = [(x_res, y_res)]
        for overview in overviews:
            resolutions.append((x_res * overview, y_res * overview))

        super().__init__(resolutions=list(reversed(resolutions)),
                         grid_mapping=grid_mapping,
                         name=path_or_url)

        self._level_datasets: List = num_levels * [None]
        # Full resolution dataset
        self._level_datasets[-1] = dataset

    @property
    def base_dataset(self) -> xr.Dataset:
        return self._level_datasets[-1]

    def get_level_dataset(self, level: Optional[int] = None) -> xr.Dataset:
        if level is None:
            level = self.num_levels - 1
        elif level < 0 or level >= self.num_levels:
            raise IndexError(f'level out of range')
        level_dataset = self._level_datasets[level]
        if level_dataset is None:
            overview_level = self.num_levels - 2 - level
            level_dataset = self.open_dataset(
                self.name,
                self.grid_mapping.tile_size,
                overview_level=overview_level
            )
            self._level_datasets[level] = level_dataset
        return level_dataset
