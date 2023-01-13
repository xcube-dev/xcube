import unittest

import numpy as np
import xarray as xr

from test.sampledata import new_test_dataset
from xcube.core.dsio import rimraf
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.xarray import DatasetAccessor


class XCubeDatasetAccessorTest(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_init(self):
        DatasetAccessor(xr.Dataset())

    def test_installed(self):
        self.assertTrue(hasattr(xr.Dataset, "xcube"))
        ds = xr.Dataset()
        self.assertTrue(hasattr(ds, "xcube"))

    def test_non_empty_cube_subset(self):
        dataset = new_cube(variables=dict(a=9, b=0.2))
        cube = dataset.xcube.cube
        self.assertIsInstance(cube, xr.Dataset)
        self.assertEqual(set(dataset.data_vars), set(cube.data_vars))
        gm = dataset.xcube.gm
        self.assertIsInstance(gm, GridMapping)

    def test_no_cube_subset(self):
        dataset = xr.Dataset(dict(a=9, b=0.2))
        cube = dataset.xcube.cube
        self.assertIsInstance(cube, xr.Dataset)
        self.assertEqual(set(), set(cube.data_vars))
        self.assertIs(None, dataset.xcube.gm)
        self.assertIs(dataset, dataset.xcube.non_cube)

    ########################################################################
    # Testing old API from here on.
    #
    # Let's quickly agree, if we should deprecate all this stuff. I guess,
    # no one uses it.
    #
    # We should only add props and methods to this accessor
    # that require a certain state to be hold.
    # Such state could be props that are expensive to recompute,
    # such as grid mappings.
    #
    # It causes too much overhead and maintenance work
    # if we continue putting any xcube function here.
    ########################################################################

    def test_new(self):
        self.assertIsInstance(xr.Dataset.xcube.new(), xr.Dataset)

    def test_select_vars(self):
        self.assertIsInstance(xr.Dataset().xcube.select_variables_subset(), xr.Dataset)

    def test_dump(self):
        self.assertIsInstance(xr.Dataset().xcube.dump(), str)

    def test_vars_to_dim(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)

        self.assertIsInstance(dataset.xcube.vars_to_dim(), xr.Dataset)

    def test_levels(self):
        dataset = new_test_dataset(
            ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04",
             "2010-01-05"],
            precipitation=0.4, temperature=275.2)
        levels = dataset.xcube.levels(spatial_tile_shape=(45, 45))
        self.assertIsInstance(levels, list)
        self.assertEqual(3, len(levels))
        self.assertTrue(
            all(isinstance(level, xr.Dataset) for level in levels))
        self.assertTrue(all("precipitation" in level for level in levels))
        self.assertTrue(all("temperature" in level for level in levels))
        self.assertEqual([(5, 180, 360), (5, 90, 180), (5, 45, 90)],
                         [level.precipitation.shape for level in levels])
        self.assertEqual([((1, 1, 1, 1, 1), (45, 45, 45, 45),
                           (45, 45, 45, 45, 45, 45, 45, 45)),
                          ((1, 1, 1, 1, 1), (45, 45), (45, 45, 45, 45)),
                          ((1, 1, 1, 1, 1), (45,), (45, 45))],
                         [level.precipitation.chunks for level in levels])


class XarrayEncodingTest(unittest.TestCase):
    """This test demonstrates how xarray uses a variable's encoding dict
    when persisting data.
    """

    zarr_path = "test.zarr"
    nc_path = "test.nc"

    def tearDown(self) -> None:
        rimraf(self.zarr_path)
        rimraf(self.nc_path)

    def test_zarr_uint8_encoding(self):
        self.assert_persistence_ok(engine="zarr",
                                   encoded_dtype=np.dtype("uint8"),
                                   decoded_dtype=np.dtype("float32"))

    def test_zarr_int16_encoding(self):
        self.assert_persistence_ok(engine="zarr",
                                   encoded_dtype=np.dtype("int16"),
                                   decoded_dtype=np.dtype("float32"))

    def test_zarr_uint32_encoding(self):
        self.assert_persistence_ok(engine="zarr",
                                   encoded_dtype=np.dtype("uint32"),
                                   decoded_dtype=np.dtype("float64"))

    def test_zarr_int64_encoding(self):
        self.assert_persistence_ok(engine="zarr",
                                   encoded_dtype=np.dtype("int64"),
                                   decoded_dtype=np.dtype("float64"))

    def test_nc_uint8_encoding(self):
        self.assert_persistence_ok(engine="netcdf4",
                                   encoded_dtype=np.dtype("uint8"),
                                   decoded_dtype=np.dtype("float32"))

    def test_nc_int16_encoding(self):
        self.assert_persistence_ok(engine="netcdf4",
                                   encoded_dtype=np.dtype("int16"),
                                   decoded_dtype=np.dtype("float32"))

    def test_nc_uint32_encoding(self):
        self.assert_persistence_ok(engine="netcdf4",
                                   encoded_dtype=np.dtype("uint32"),
                                   decoded_dtype=np.dtype("float64"))

    def test_nc_int64_encoding(self):
        self.assert_persistence_ok(engine="netcdf4",
                                   encoded_dtype=np.dtype("int64"),
                                   decoded_dtype=np.dtype("float64"))

    def assert_persistence_ok(self,
                              engine: str,
                              encoded_dtype: np.dtype,
                              decoded_dtype: np.dtype,
                              encoded_fill_value: int = 2,
                              decoded_fill_value: float = np.nan):
        a = xr.DataArray(np.array([1, np.nan, 3],
                                  dtype=np.float64),
                         dims="x")
        a.encoding.update(_FillValue=encoded_fill_value,
                          dtype=encoded_dtype)
        ds = xr.Dataset(dict(A=a))

        if engine == "zarr":
            path = self.zarr_path
            ds.to_zarr(path, mode="w")
        else:
            path = self.nc_path
            ds.to_netcdf(path, mode="w")

        # No kwargs
        ds = xr.open_dataset(path, engine=engine)
        np.testing.assert_equal(ds.A.values,
                                np.array([1, decoded_fill_value, 3]))
        self.assertEqual(decoded_dtype, ds.A.dtype)

        # decode_cf=True
        ds = xr.open_dataset(path, engine=engine, decode_cf=True)
        np.testing.assert_equal(ds.A.values,
                                np.array([1, decoded_fill_value, 3]))
        self.assertEqual(decoded_dtype, ds.A.dtype)

        # decode_cf=False
        ds = xr.open_dataset(path, engine=engine, decode_cf=False)
        np.testing.assert_equal(ds.A.values,
                                np.array([1, encoded_fill_value, 3]))
        self.assertEqual(encoded_dtype, ds.A.dtype)
