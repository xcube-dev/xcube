import datetime
import numbers
import warnings
from typing import Optional, Sequence, Any, Tuple

import pandas as pd
import pyproj

from xcube.core.store import DataStorePool
from xcube.core.store import DatasetDescriptor
from xcube.core.store import VariableDescriptor
from xcube.util.assertions import assert_condition
from xcube.util.assertions import assert_instance
from .config import CubeConfig
from .describer import DatasetsDescriber
from .request import CubeGeneratorRequest
from .response import CubeInfo


class CubeInformant:
    def __init__(self,
                 request: CubeGeneratorRequest,
                 store_pool: DataStorePool = None):
        assert_instance(request, CubeGeneratorRequest, name='request')
        if store_pool is not None:
            assert_instance(store_pool, DataStorePool, name='store_pool')
        self._request: CubeGeneratorRequest = request
        self._store_pool: Optional[DataStorePool] = store_pool
        self._dataset_descriptors: Optional[Sequence[DatasetDescriptor]] = None

    @property
    def input_dataset_descriptors(self) -> Sequence[DatasetDescriptor]:
        """Get dataset descriptors for inputs, lazily."""
        if self._dataset_descriptors is None:
            dataset_describer = DatasetsDescriber(self._request.input_configs,
                                                  store_pool=self._store_pool)
            dataset_descriptors = dataset_describer.describe_datasets()
            if len(dataset_descriptors) > 1:
                warnings.warn('Only the first input will be recognised.')
            self._dataset_descriptors = dataset_descriptors
        return self._dataset_descriptors

    @property
    def first_input_dataset_descriptor(self) -> DatasetDescriptor:
        """Get dataset descriptor of first input, lazily."""
        return self.input_dataset_descriptors[0]

    @property
    def effective_cube_config(self) -> CubeConfig:
        cube_config, _ = self.compute_effective_cube_config()
        return cube_config

    def generate(self) -> CubeInfo:
        cube_config, is_geo_crs = self.compute_effective_cube_config()

        x_min, y_min, x_max, y_max = cube_config.bbox
        spatial_res = cube_config.spatial_res

        width = round((x_max - x_min) / spatial_res)
        height = round((y_max - y_min) / spatial_res)
        width = 2 if width < 2 else width
        height = 2 if height < 2 else height

        num_tiles_x = 1
        num_tiles_y = 1
        tile_width = width
        tile_height = height

        tile_size = cube_config.tile_size
        if tile_size is not None:
            try:
                tile_width, tile_height = tile_size
            except (TypeError, ValueError):
                raise ValueError('tile_size must be a tuple (tile_width, tile_height)')
            assert_instance(tile_width, int, 'tile_width')
            assert_instance(tile_height, int, 'tile_height')

            # TODO: this must be made common store logic
            if width > 1.5 * tile_width:
                num_tiles_x = _idiv(width, tile_width)
                width = num_tiles_x * tile_width

            # TODO: this must be made common store logic
            if height > 1.5 * tile_height:
                num_tiles_y = _idiv(height, tile_height)
                height = num_tiles_y * tile_height

        time_range = cube_config.time_range
        time_period = cube_config.time_period
        try:
            date_range = pd.date_range(start=time_range[0], end=time_range[1], freq=time_period)
        except ValueError as e:
            raise ValueError(f'invalid time_range or time_period: {e}') from e

        variable_names = cube_config.variable_names

        num_times = len(date_range)
        num_variables = len(variable_names)
        num_requests = num_variables * num_times * num_tiles_x * num_tiles_y
        # TODO: get original data types from dataset descriptors
        num_bytes_per_pixel = 4
        num_bytes = num_variables * num_times * (height * width * num_bytes_per_pixel)

        x_name, y_name = ('lon', 'lat') if is_geo_crs else ('x', 'y')

        data_id = self._request.output_config.data_id or 'unnamed'
        # TODO: get original variable descriptors from input dataset descriptors
        data_vars = {name: VariableDescriptor(name, dtype='float32', dims=('time', y_name, x_name))
                     for name in variable_names}
        dims = {'time': num_times, y_name: height, x_name: width}
        dataset_descriptor = DatasetDescriptor(data_id,
                                               crs=cube_config.crs,
                                               bbox=cube_config.bbox,
                                               spatial_res=cube_config.spatial_res,
                                               time_range=cube_config.time_range,
                                               time_period=cube_config.time_period,
                                               dims=dims,
                                               data_vars=data_vars)
        size_estimation = dict(image_size=[width, height],
                               tile_size=[tile_width, tile_height],
                               num_variables=num_variables,
                               num_tiles=[num_tiles_x, num_tiles_y],
                               num_requests=num_requests,
                               num_bytes=num_bytes)

        return CubeInfo(dataset_descriptor=dataset_descriptor,
                        size_estimation=size_estimation)

    def compute_effective_cube_config(self) -> Tuple[CubeConfig, bool]:
        cube_config = self._request.cube_config

        crs = cube_config.crs
        if crs is None:
            crs = self.first_input_dataset_descriptor.crs
            if crs is None:
                crs = 'WGS84'
        is_geo_crs = pyproj.crs.CRS.from_string(crs).is_geographic

        bbox = cube_config.bbox
        if bbox is None:
            bbox = self.first_input_dataset_descriptor.bbox
        try:
            x1, y1, x2, y2 = bbox
        except (TypeError, ValueError):
            raise ValueError('bbox must be a tuple (x1, y1, x2, y2)')
        assert_instance(x1, numbers.Number, 'x1 of bbox')
        assert_instance(y1, numbers.Number, 'y1 of bbox')
        assert_instance(x2, numbers.Number, 'x2 of bbox')
        assert_instance(y2, numbers.Number, 'y2 of bbox')
        if is_geo_crs and x2 < x1:
            x2 += 360
        bbox = x1, y1, x2, y2

        spatial_res = cube_config.spatial_res
        if spatial_res is None:
            spatial_res = self.first_input_dataset_descriptor.spatial_res
        assert_instance(spatial_res, numbers.Number, 'spatial_res')
        assert_condition(spatial_res > 0, 'spatial_res must be positive')

        tile_size = cube_config.tile_size
        if tile_size is not None:
            try:
                tile_width, tile_height = tile_size
            except (TypeError, ValueError):
                raise ValueError('tile_size must be a tuple (tile_width, tile_height)')
            tile_size = tile_width, tile_height

        time_range = cube_config.time_range
        if time_range is None:
            time_range = None, None
        start_ts, end_ts = _parse_time_range(time_range)
        if start_ts is None or end_ts is None:
            default_time_range = self.first_input_dataset_descriptor.time_range
            if default_time_range is None:
                default_time_range = None, None
            default_start_ts, default_end_ts = _parse_time_range(default_time_range)
            if start_ts is None:
                start_ts = default_start_ts
                if start_ts is None:
                    start_ts = pd.Timestamp(datetime.date.today())
            if end_ts is None:
                end_ts = default_end_ts
                if end_ts is None:
                    end_ts = pd.Timestamp(datetime.date.today()) + pd.Timedelta('1D')
        time_range = start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")

        time_period = cube_config.time_period
        if time_period is None:
            time_period = self.first_input_dataset_descriptor.time_period
            if time_period is None:
                time_period = '1D'
        assert_instance(time_period, str, 'time_period')

        variable_names = cube_config.variable_names
        if variable_names is None:
            variable_names = list(self.first_input_dataset_descriptor.data_vars.keys())

        return CubeConfig(variable_names=variable_names,
                          crs=crs,
                          bbox=bbox,
                          spatial_res=spatial_res,
                          tile_size=tile_size,
                          time_range=time_range,
                          time_period=time_period), is_geo_crs


def _idiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _parse_time_range(time_range: Any) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    try:
        start_date, end_date = time_range
    except (TypeError, ValueError):
        raise ValueError('time_range must be a tuple (start_date, end_date)')
    try:
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
        if end_date is not None:
            end_date = pd.Timestamp(end_date)
    except ValueError as e:
        raise ValueError(f'time_range is invalid: {e}') from e
    return start_date, end_date
