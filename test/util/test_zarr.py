import os.path
import shutil
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .diagnosticstore import DiagnosticStore, logging_observer


class ZarrStoreTest(unittest.TestCase):
    CUBE_PATH = 'store-test-cube.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)

    def test_local(self):
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        cube = chunk_cube(cube, dict(time=1, lat=90, lon=90))
        cube.to_zarr(self.CUBE_PATH)
        cube.close()

        diag_store = DiagnosticStore(zarr.DirectoryStore(self.CUBE_PATH),
                                     logging_observer(log_path='local-cube.log'))
        xr.open_zarr(diag_store)

    @unittest.skipUnless(False, 'is enabled')
    def test_remote(self):
        import s3fs
        endpoint_url = "http://obs.eu-de.otc.t-systems.com"
        s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(endpoint_url=endpoint_url))
        s3_store = s3fs.S3Map(root="cyanoalert/cyanoalert-olci-lswe-l2c-v1.zarr", s3=s3, check=False)
        diag_store = DiagnosticStore(s3_store, logging_observer(log_path='remote-cube.log'))
        xr.open_zarr(diag_store)


class ZarrAppendTest(unittest.TestCase):
    CUBE_PATH = 'append-test-cube.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)

    def test_append_possible_with_zarr(self):

        times = []
        for m in range(1, 4):
            for d in range(1, 10):
                times.append(f'2019-0{m}-0{d}T00:00:00')

        for i_time in range(len(times)):
            slice = new_cube(time_periods=1, time_start=times[i_time],
                             variables=dict(precipitation=0.1 * i_time,
                                            temperature=270 + 0.5 * i_time,
                                            soil_moisture=0.2 * i_time))
            slice = chunk_cube(slice, dict(time=1, lat=90, lon=90))
            slice.to_zarr(self.CUBE_PATH, mode='a', append_dim='time')
            unchunk_time_vars(self.CUBE_PATH)

        with xr.open_zarr(self.CUBE_PATH) as cube:
            self.assertEqual(len(times), cube.time.size)
            self.assertEqual(None, cube.time.chunks)
            actual = cube.time.values
            expected = np.array(['2019-01-01T12:00', '2019-01-02T12:00',
                                 '2019-01-03T12:00', '2019-01-04T12:00',
                                 '2019-01-05T12:00', '2019-01-06T12:00',
                                 '2019-01-07T12:00', '2019-01-08T12:00',
                                 '2019-01-09T12:00', '2019-02-01T12:00',
                                 '2019-02-02T12:00', '2019-02-03T12:00',
                                 '2019-02-04T12:00', '2019-02-05T12:00',
                                 '2019-02-06T12:00', '2019-02-07T12:00',
                                 '2019-02-08T12:00', '2019-02-09T12:00',
                                 '2019-03-01T12:00', '2019-03-02T12:00',
                                 '2019-03-03T12:00', '2019-03-04T12:00',
                                 '2019-03-05T12:00', '2019-03-06T12:00',
                                 '2019-03-07T12:00', '2019-03-08T12:00',
                                 '2019-03-09T12:00'], dtype=actual.dtype)
            np.testing.assert_equal(actual, expected)


class ZarrInsertTest(unittest.TestCase):
    CUBE_PATH = 'insert-test-cube.zarr'
    SLICE_PATH = 'slice.zarr'

    def setUp(self) -> None:
        rimraf(self.CUBE_PATH)
        rimraf(self.SLICE_PATH)
        cube = new_cube(time_periods=10, time_start='2019-01-01',
                        variables=dict(precipitation=0.1,
                                       temperature=270.5,
                                       soil_moisture=0.2))
        cube = chunk_cube(cube, dict(time=1, lat=90, lon=90))
        cube.to_zarr(self.CUBE_PATH)
        cube.close()

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH)
        rimraf(self.SLICE_PATH)

    def test_insert_possible_with_zarr(self):

        with xr.open_zarr(self.CUBE_PATH) as cube:
            time_var_names = [var_name for var_name in cube.variables if cube[var_name].dims[0] == 'time']

        insert_index = 8
        slice = new_cube(time_periods=1, time_start='2019-01-08T16:30:00',
                         variables=dict(precipitation=0.2,
                                        temperature=275.8,
                                        soil_moisture=0.4))

        slice = chunk_cube(slice, dict(time=1, lat=90, lon=90))
        slice.to_zarr(self.SLICE_PATH)
        slice_root_group = zarr.open(self.SLICE_PATH, mode='r')
        slice_arrays = dict(slice_root_group.arrays())

        cube_root_group = zarr.open(self.CUBE_PATH, mode='r+')
        for var_name, var_array in cube_root_group.arrays():
            if var_name in time_var_names:
                slice_array = slice_arrays[var_name]
                # Add one empty time step
                empty = zarr.creation.empty(slice_array.shape, dtype=var_array.dtype)
                var_array.append(empty, axis=0)
                # Shift contents
                var_array[insert_index + 1:, ...] = var_array[insert_index:-1, ...]
                # Insert slice
                var_array[insert_index, ...] = slice_array[0]

        unchunk_time_vars(self.CUBE_PATH)

        with xr.open_zarr(self.CUBE_PATH) as cube:
            self.assertEqual(11, cube.time.size)
            self.assertEqual(None, cube.time.chunks)
            actual = cube.time.values
            expected = np.array(['2019-01-01T12:00',
                                 '2019-01-02T12:00',
                                 '2019-01-03T12:00',
                                 '2019-01-04T12:00',
                                 '2019-01-05T12:00',
                                 '2019-01-06T12:00',
                                 '2019-01-07T12:00',
                                 '2019-01-08T12:00',
                                 '2019-01-09T04:30',
                                 '2019-01-09T12:00',
                                 '2019-01-10T12:00'], dtype=actual.dtype)
            np.testing.assert_equal(actual, expected)


def chunk_cube(cube: xr.Dataset, chunk_sizes):
    cube = cube.chunk(chunks=chunk_sizes)
    for var_name in cube.variables:
        var = cube[var_name]
        if var.chunks:
            chunks = tuple(chunk_sizes.get(var.dims[i], var.shape[i]) for i in range(var.ndim))
            var.encoding.update(dict(chunks=chunks))
    return cube


_TIME_DTYPE = "datetime64[s]"
_TIME_UNITS = "seconds since 2019-01-01T00:00:00"
_TIME_CALENDAR = "proleptic_gregorian"


def new_cube(title="Test Cube",
             width=360,
             height=180,
             spatial_res=1.0,
             lon_start=-180.0,
             lat_start=-90.0,
             time_periods=5,
             time_freq="D",
             time_start='2010-01-01T00:00:00',
             drop_bounds=False,
             variables=None):
    """
    Create a new empty cube. Useful for testing.

    :param title: A title.
    :param width: Horizontal number of grid cells
    :param height: Vertical number of grid cells
    :param spatial_res: Spatial resolution in degrees
    :param lon_start: Minimum longitude value
    :param lat_start: Minimum latitude value
    :param time_periods: Number of time steps
    :param time_freq: Duration of each time step
    :param time_start: First time value
    :param drop_bounds: If True, coordinate bounds variables are not created.
    :param variables: Dictionary of data variables to be added.
    :return: A cube instance
    """
    lon_end = lon_start + width * spatial_res
    lat_end = lat_start + height * spatial_res
    if width < 0 or height < 0 or spatial_res <= 0.0:
        raise ValueError()
    if lon_start < -180. or lon_end > 180. or lat_start < -90. or lat_end > 90.:
        raise ValueError()
    if time_periods < 0:
        raise ValueError()

    lon_data = np.linspace(lon_start + 0.5 * spatial_res, lon_end - 0.5 * spatial_res, width)
    lon = xr.DataArray(lon_data, dims="lon")
    lon.attrs["units"] = "degrees_east"

    lat_data = np.linspace(lat_start + 0.5 * spatial_res, lat_end - 0.5 * spatial_res, height)
    lat = xr.DataArray(lat_data, dims="lat")
    lat.attrs["units"] = "degrees_north"

    time_data_2 = pd.date_range(start=time_start, periods=time_periods + 1, freq=time_freq).values
    time_data_2 = time_data_2.astype(dtype=_TIME_DTYPE)
    time_delta = time_data_2[1] - time_data_2[0]
    time_data = time_data_2[0:-1] + time_delta // 2
    time = xr.DataArray(time_data, dims="time")
    time.encoding["units"] = _TIME_UNITS
    time.encoding["calendar"] = _TIME_CALENDAR

    time_data_2 = pd.date_range(time_start, periods=time_periods + 1, freq=time_freq)

    coords = dict(lon=lon, lat=lat, time=time)
    if not drop_bounds:
        lon_bnds_data = np.zeros((width, 2), dtype=np.float64)
        lon_bnds_data[:, 0] = np.linspace(lon_start, lon_end - spatial_res, width)
        lon_bnds_data[:, 1] = np.linspace(lon_start + spatial_res, lon_end, width)
        lon_bnds = xr.DataArray(lon_bnds_data, dims=("lon", "bnds"))
        lon_bnds.attrs["units"] = "degrees_east"

        lat_bnds_data = np.zeros((height, 2), dtype=np.float64)
        lat_bnds_data[:, 0] = np.linspace(lat_start, lat_end - spatial_res, height)
        lat_bnds_data[:, 1] = np.linspace(lat_start + spatial_res, lat_end, height)
        lat_bnds = xr.DataArray(lat_bnds_data, dims=("lat", "bnds"))
        lat_bnds.attrs["units"] = "degrees_north"

        time_bnds_data = np.zeros((time_periods, 2), dtype="datetime64[ns]")
        time_bnds_data[:, 0] = time_data_2[:-1]
        time_bnds_data[:, 1] = time_data_2[1:]
        time_bnds = xr.DataArray(time_bnds_data, dims=("time", "bnds"))
        time_bnds.encoding["units"] = _TIME_UNITS
        time_bnds.encoding["calendar"] = _TIME_CALENDAR

        lon.attrs["bounds"] = "lon_bnds"
        lat.attrs["bounds"] = "lat_bnds"
        time.attrs["bounds"] = "time_bnds"

        coords.update(dict(lon_bnds=lon_bnds, lat_bnds=lat_bnds, time_bnds=time_bnds))

    attrs = {
        "Conventions": "CF-1.7",
        "title": title,
        "time_coverage_start": str(time_data_2[0]),
        "time_coverage_end": str(time_data_2[-1]),
        "geospatial_lon_min": lon_start,
        "geospatial_lon_max": lon_end,
        "geospatial_lon_units": "degrees_east",
        "geospatial_lat_min": lat_start,
        "geospatial_lat_max": lat_end,
        "geospatial_lat_units": "degrees_north",
    }

    # TODO (forman): allow variable values to be expressions so values will be computed from coords using numexpr

    data_vars = {}
    if variables:
        dims = ("time", "lat", "lon")
        shape = (time_periods, height, width)
        size = time_periods * height * width
        for var_name, data in variables.items():
            if isinstance(data, xr.DataArray):
                data_vars[var_name] = data
            elif isinstance(data, int) or isinstance(data, float) or isinstance(data, bool):
                data_vars[var_name] = xr.DataArray(np.full(shape, data), dims=dims)
            elif data is None:
                data_vars[var_name] = xr.DataArray(np.random.uniform(0.0, 1.0, size).reshape(shape), dims=dims)
            else:
                data_vars[var_name] = xr.DataArray(data, dims=dims)

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def rimraf(path):
    """
    The UNIX command `rm -rf`.
    Recursively remove directory or single file.

    :param path:  directory or single file
    """
    if os.path.isdir(path):
        try:
            shutil.rmtree(path, ignore_errors=False)
        except OSError:
            warnings.warn(f"failed to remove file {path}")
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            warnings.warn(f"failed to remove file {path}")
            pass


def unchunk_time_vars(cube_path: str):
    with xr.open_zarr(cube_path) as dataset:
        coord_var_names = [var_name for var_name in dataset.coords if 'time' in dataset[var_name].dims]
    for coord_var_name in coord_var_names:
        coord_var_path = os.path.join(cube_path, coord_var_name)
        coord_var_array = zarr.convenience.open_array(coord_var_path, 'r+')
        # Fully load data and attrs so we no longer depend on files
        data = np.array(coord_var_array)
        attrs = coord_var_array.attrs.asdict()
        # Save array data
        zarr.convenience.save_array(coord_var_path, data, chunks=False, fill_value=coord_var_array.fill_value)
        # zarr.convenience.save_array() does not seem save user attributes (file ".zattrs" not written),
        # therefore we must modify attrs explicitly:
        coord_var_array = zarr.convenience.open_array(coord_var_path, 'r+')
        coord_var_array.attrs.update(attrs)
