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
from numbers import Number
from typing import Any, Mapping, MutableMapping, Sequence, Hashable, Type, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from xcube.constants import LOG
from xcube.core.mldataset import MultiLevelDataset
from xcube.core.schema import get_dataset_xy_var_names
from xcube.util.cache import Cache
from xcube.util.perf import measure_time_cm
from xcube.util.tiledimage import SourceArrayImage
from xcube.util.tiledimage import ColorMappedRgbaImage
from xcube.util.tiledimage import DEFAULT_COLOR_MAP_NAME
from xcube.util.tiledimage import DEFAULT_COLOR_MAP_VALUE_RANGE
from xcube.util.tiledimage import DirectRgbaImage
from xcube.util.tiledimage import Tile
from xcube.util.tiledimage import TiledImage
from xcube.util.tiledimage import NormalizeArrayImage


def get_ml_dataset_tile(
        ml_dataset: MultiLevelDataset,
        var_name: str,
        x: int,
        y: int,
        z: int,
        labels: Mapping[str, Any] = None,
        labels_are_indices: bool = False,
        cmap_name: Union[str, Tuple[Optional[str]]] = None,
        cmap_range: Union[Tuple[float, float],
                          Tuple[Tuple[float, float],
                                Tuple[float, float],
                                Tuple[float, float]]] = None,
        image_cache: MutableMapping[str, TiledImage] = None,
        tile_cache: Cache = None,
        trace_perf: bool = False,
        exception_type: Type[Exception] = ValueError
) -> Tile:
    labels = labels or {}
    ds_cache_id = hex(id(ml_dataset))
    image_id = '-'.join(map(str, [ds_cache_id,
                                  z,
                                  ml_dataset.ds_id,
                                  var_name,
                                  cmap_name,
                                  cmap_range]
                            + [f'{dim_name}={dim_value}'
                               for dim_name, dim_value in labels.items()]))

    measure_time = measure_time_cm(logger=LOG, disabled=not trace_perf)

    if image_cache and image_id in image_cache:
        image = image_cache[image_id]
    else:
        with measure_time() as measured_time:
            if var_name == 'rgb':
                image = new_rgb_image(ml_dataset,
                                      image_id,
                                      cmap_name,
                                      cmap_range,
                                      z,
                                      labels,
                                      labels_are_indices,
                                      tile_cache,
                                      trace_perf,
                                      exception_type)
            else:
                image = new_color_mapped_image(ml_dataset,
                                               image_id,
                                               var_name,
                                               cmap_name,
                                               cmap_range,
                                               z,
                                               labels,
                                               labels_are_indices,
                                               tile_cache,
                                               trace_perf,
                                               exception_type)

        if image_cache:
            image_cache[image_id] = image

        if trace_perf:
            tile_grid = ml_dataset.tile_grid
            LOG.info(
                f'Created tiled image {image_id!r} of size {image.size}.\n'
                f'Took {measured_time.duration:.2f} seconds. Tile grid:\n'
                f'  num_levels: {tile_grid.num_levels}\n'
                f'  num_level_zero_tiles: {tile_grid.get_num_tiles(0)}\n'
                f'  tile_size: {tile_grid.tile_size}\n'
                f'  extent: {tile_grid.extent}\n'
                f'  is_j_axis_up: {tile_grid.is_j_axis_up}'
            )

    if trace_perf:
        LOG.info(f'>>> tile {image_id}/{z}/{y}/{x}')

    with measure_time() as measured_time:
        tile = image.get_tile(x, y)

    if trace_perf:
        LOG.info(f'<<< tile {image_id}/{z}/{y}/{x}:'
                 f' took ' + '%.2f seconds' % measured_time.duration)

    return tile


def new_rgb_image(ml_dataset: MultiLevelDataset,
                  image_id: str,
                  var_names: Tuple[str, ...],
                  norm_ranges: Tuple[Number, Number],
                  z: int,
                  labels,
                  labels_are_indices,
                  tile_cache,
                  trace_perf,
                  exception_type):
    tile_grid = ml_dataset.tile_grid
    images = []
    for i in range(3):
        var_name = var_names[i]
        array, _, _ = \
            _get_var_2d_array_and_mask_info(
                ml_dataset, var_name, z, labels,
                labels_are_indices,
                exception_type
            )

        image = SourceArrayImage(array,
                                 image_id=f'ai-{image_id}',
                                 tile_size=tile_grid.tile_size,
                                 flip_y=tile_grid.is_j_axis_up,
                                 trace_perf=trace_perf)
        image = NormalizeArrayImage(image,
                                    image_id=f'tai-{image_id}',
                                    norm_range=norm_ranges[i],
                                    trace_perf=trace_perf)
        images.append(image)

    return DirectRgbaImage((images[0], images[1], images[2]),
                           image_id=f'drgba-{image_id}',
                           encode=True,
                           format='PNG',
                           tile_cache=tile_cache,
                           trace_perf=trace_perf)


def new_color_mapped_image(ml_dataset: MultiLevelDataset,
                           image_id: str,
                           var_name: str,
                           cmap_name: str,
                           cmap_range: Tuple[float, float],
                           z: int,
                           labels: Dict[str, Any],
                           labels_are_indices: bool,
                           tile_cache,
                           trace_perf,
                           exception_type: Type[Exception]):
    array, valid_range, var = \
        _get_var_2d_array_and_mask_info(
            ml_dataset, var_name, z, labels,
            labels_are_indices,
            exception_type
        )
    cmap_name, cmap_range = get_var_cmap_params(var,
                                                cmap_name,
                                                cmap_range,
                                                valid_range)
    tile_grid = ml_dataset.tile_grid
    image = SourceArrayImage(array,
                             image_id=f'ai-{image_id}',
                             tile_size=tile_grid.tile_size,
                             flip_y=tile_grid.is_j_axis_up,
                             trace_perf=trace_perf)
    image = NormalizeArrayImage(image,
                                image_id=f'tai-{image_id}',
                                norm_range=cmap_range,
                                trace_perf=trace_perf)
    if cmap_name is None and cmap_range is None:
        # TODO: Return a ColorMappedRawImage(...) whose tiles
        #   are binary, compressed numpy arrays
        return image
    return ColorMappedRgbaImage(image,
                                image_id=f'cmrgbai-{image_id}',
                                cmap_name=cmap_name,
                                encode=True,
                                format='PNG',
                                tile_cache=tile_cache,
                                trace_perf=trace_perf)


def _get_var_2d_array_and_mask_info(ml_dataset: MultiLevelDataset,
                                    var_name: str,
                                    z: int,
                                    labels: Dict[str, Any],
                                    labels_are_indices: bool,
                                    exception_type: Type[Exception]):
    dataset = ml_dataset.get_dataset(ml_dataset.num_levels - 1 - z)
    var = dataset[var_name]
    valid_range = get_var_valid_range(var)
    array = _get_var_2d_array(var,
                              labels,
                              labels_are_indices,
                              exception_type,
                              ml_dataset.ds_id)
    return array, valid_range, var


def _get_var_2d_array(var: xr.DataArray,
                      labels: Dict[str, Any],
                      labels_are_indices: bool,
                      exception_type: Type[Exception],
                      ds_id: str) -> xr.DataArray:
    # Make sure we work with 2D image arrays only
    if var.ndim == 2:
        assert len(labels) == 0
        array = var
    elif var.ndim > 2:
        assert len(labels) == var.ndim - 2
        if labels_are_indices:
            array = var.isel(**labels)
        else:
            array = var.sel(method='nearest', **labels)
    else:
        raise exception_type(f'Variable "{var.name}" of dataset "{ds_id}" '
                             'must be an N-D Dataset with N >= 2, '
                             f'but "{var.name}" is only {var.ndim}-D')
    return array


def _ensure_time_compatible(var: xr.DataArray,
                            labels: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure that labels['time'] is timezone-naive, if necessary.
    If var has a 'time' dimension of type datetime64 and labels has a 'time'
    key with a timezone-aware value, return a modified labels dictionary with
    a timezone-naive time value. Otherwise return the original labels.
    """
    if _has_datetime64_time(var) and \
       'time' in labels and pd.Timestamp(labels['time']).tzinfo is not None:
        naive_time = pd.Timestamp(labels['time']).tz_convert(None)
        return dict(labels, time=naive_time)
    else:
        return labels


def _has_datetime64_time(var: xr.DataArray) -> bool:
    """Report whether var has a time dimension with type datetime64"""
    return 'time' in var.dims and \
           hasattr(var['time'], 'dtype') and \
           hasattr(var['time'].dtype, 'type') and \
           var['time'].dtype.type is np.datetime64


def get_var_cmap_params(var: xr.DataArray,
                        cmap_name: Optional[str],
                        cmap_range: Tuple[Optional[float], Optional[float]],
                        valid_range: Optional[Tuple[float, float]]) \
        -> Tuple[str, Tuple[float, float]]:
    if cmap_name is None:
        cmap_name = var.attrs.get('color_bar_name')
        if cmap_name is None:
            cmap_name = DEFAULT_COLOR_MAP_NAME
    cmap_vmin, cmap_vmax = cmap_range
    if cmap_vmin is None:
        cmap_vmin = var.attrs.get('color_value_min')
        if cmap_vmin is None and valid_range is not None:
            cmap_vmin = valid_range[0]
        if cmap_vmin is None:
            cmap_vmin = DEFAULT_COLOR_MAP_VALUE_RANGE[0]
    if cmap_vmax is None:
        cmap_vmax = var.attrs.get('color_value_max')
        if cmap_vmax is None and valid_range is not None:
            cmap_vmax = valid_range[1]
        if cmap_vmax is None:
            cmap_vmax = DEFAULT_COLOR_MAP_VALUE_RANGE[1]
    return cmap_name, (cmap_vmin, cmap_vmax)


def get_var_valid_range(var: xr.DataArray) -> Optional[Tuple[float, float]]:
    valid_min = None
    valid_max = None
    valid_range = var.attrs.get('valid_range')
    if valid_range:
        try:
            valid_min, valid_max = map(float, valid_range)
        except (TypeError, ValueError):
            pass
    if valid_min is None:
        valid_min = var.attrs.get('valid_min')
    if valid_max is None:
        valid_max = var.attrs.get('valid_max')
    if valid_min is None and valid_max is None:
        valid_range = None
    elif valid_min is not None and valid_max is not None:
        valid_range = valid_min, valid_max
    elif valid_min is None:
        valid_range = -np.inf, valid_max
    else:
        valid_range = valid_min, +np.inf
    return valid_range


def parse_non_spatial_labels(
        raw_labels: Mapping[str, str],
        dims: Sequence[Hashable],
        coords: Mapping[Hashable, xr.DataArray],
        allow_slices: bool = False,
        exception_type: type = ValueError,
        var: xr.DataArray = None
) -> Mapping[str, Any]:
    xy_var_names = get_dataset_xy_var_names(coords, must_exist=False)
    if xy_var_names is None:
        raise exception_type(f'missing spatial coordinates')
    xy_dims = set(coords[xy_var_name].dims[0] for xy_var_name in xy_var_names)

    # noinspection PyShadowingNames
    def to_datetime(datetime_str: str, dim_var: xr.DataArray):
        if datetime_str == 'current':
            return dim_var[-1]
        else:
            return pd.to_datetime(datetime_str)

    parsed_labels = {}
    for dim in dims:
        if dim in xy_dims:
            continue
        dim_var = coords[dim]
        label_str = raw_labels.get(str(dim))
        try:
            if label_str is None:
                label = dim_var[0].values
            elif label_str == 'current':
                label = dim_var[-1].values
            else:
                if '/' in label_str:
                    label_strs = tuple(label_str.split('/', maxsplit=1))
                else:
                    label_strs = (label_str,)
                if np.issubdtype(dim_var.dtype, np.floating):
                    labels = tuple(map(float, label_strs))
                elif np.issubdtype(dim_var.dtype, np.integer):
                    labels = tuple(map(int, label_strs))
                elif np.issubdtype(dim_var.dtype, np.datetime64):
                    labels = tuple(to_datetime(label, dim_var)
                                   for label in label_strs)
                else:
                    raise exception_type(f'unable to parse value'
                                         f' {label_str!r}'
                                         f' into a {dim_var.dtype!r}')
                if len(labels) == 1:
                    label = labels[0]
                else:
                    if allow_slices:
                        # noinspection PyTypeChecker
                        label = slice(labels[0], labels[1])
                    elif np.issubdtype(dim_var.dtype, np.integer):
                        label = labels[0] + (labels[1] - labels[0]) // 2
                    else:
                        label = labels[0] + (labels[1] - labels[0]) / 2
            parsed_labels[str(dim)] = label
        except ValueError as e:
            raise exception_type(f'{label_str!r} is not a valid'
                                 f' value for dimension {dim!r}') from e

    if var is not None:
        return _ensure_time_compatible(var, parsed_labels)
    else:
        return parsed_labels
