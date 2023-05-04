import unittest
from typing import Tuple

import numpy as np
import pyproj
import pytest
import xarray as xr

from xcube.core.gridmapping import GridMapping, CRS_CRS84
from xcube.core.gridmapping.cfconv import find_grid_mapping_for_data_var

CRS_WGS84 = pyproj.crs.CRS(4326)
CRS_UTM_33N = pyproj.crs.CRS(32633)

CRS_ROTATED_POLE = pyproj.crs.CRS.from_cf(
    dict(grid_mapping_name="rotated_latitude_longitude",
         grid_north_pole_latitude=32.5,
         grid_north_pole_longitude=170.))


class FindGridMappingForVarTest(unittest.TestCase):
    w = 20
    h = 10

    def assert_grid_mapping_tuple_tuple_ok(
            self,
            actual_gmt,
            expected_crs: pyproj.CRS,
            expected_gm_name: str,
            expected_xy_names: Tuple[str, str],
            expected_xy_dims: Tuple[str, str],
            expected_xy_sizes: Tuple[int, int]
    ):
        self.assertIsInstance(actual_gmt, tuple)
        self.assertEqual(3, len(actual_gmt))
        crs, gm_name, xy_coords = actual_gmt
        self.assertEqual(expected_crs.to_cf(), crs.to_cf())
        self.assertEqual(expected_gm_name, gm_name)
        self.assertIsInstance(xy_coords, tuple)
        self.assertEqual(2, len(xy_coords))
        x, y = xy_coords
        self.assertIsInstance(x, xr.DataArray)
        self.assertIsInstance(y, xr.DataArray)
        self.assertEqual(1, x.ndim)
        self.assertEqual(1, y.ndim)
        self.assertEqual(expected_xy_dims, (x.dims[0], y.dims[0]))
        self.assertEqual(expected_xy_names, (x.name, y.name))
        self.assertEqual(expected_xy_sizes, (x.size, y.size))

    def new_data_var(self,
                     shape=None,
                     dims=None,
                     attrs=None) -> xr.DataArray:
        return xr.DataArray(
            np.zeros(shape or (1, self.h, self.w), dtype=np.float32),
            dims=dims or ("time", "y", "x"),
            attrs=attrs
        )

    def new_x_coord_var(self,
                        dim=None,
                        attrs=None) -> xr.DataArray:
        return xr.DataArray(
            np.linspace(10000, 10000 + 10 * self.w, self.w),
            dims=dim or "x",
            attrs=attrs
        )

    def new_y_coord_var(self,
                        dim=None,
                        attrs=None) -> xr.DataArray:
        return xr.DataArray(
            np.linspace(10000, 10000 + 10 * self.h, self.h),
            dims=dim or "y",
            attrs=attrs
        )

    def new_crs_var(self, crs) -> xr.DataArray:
        return xr.DataArray(0, attrs=crs.to_cf())

    def test_with_gm_var_and_named_coords(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs: a b"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "a": self.new_x_coord_var(),
                "b": self.new_y_coord_var(),
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_UTM_33N,
            "transverse_mercator",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_fails_with_invalid_coords(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs: a b"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        with pytest.raises(ValueError,
                           match="invalid coordinates in"
                                 " grid mapping value 'crs: a b'"):
            find_grid_mapping_for_data_var(ds, "sst")

    def test_with_gm_var_fails_with_invalid_crs(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": xr.DataArray(0, attrs={"bibo": 12})
            },
            coords={
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        with pytest.raises(ValueError,
                           match="variable 'crs' is not"
                                 " a valid grid mapping"):
            find_grid_mapping_for_data_var(ds, "sst")

    def test_with_gm_var_fails_with_invalid_gm_var(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "spatial_ref": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        with pytest.raises(ValueError,
                           match="grid mapping variable 'crs'"
                                 " not found in dataset"):
            find_grid_mapping_for_data_var(ds, "sst")

    def test_with_gm_var_and_standard_name_lat_lon(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_WGS84)
            },
            coords={
                "a": self.new_x_coord_var(
                    attrs={"standard_name": "longitude"}
                ),
                "b": self.new_y_coord_var(
                    attrs={"standard_name": "latitude"}
                ),
                "lon": self.new_x_coord_var(),
                "lat": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_WGS84,
            "latitude_longitude",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_and_standard_name_rot_lat_lon(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_ROTATED_POLE)
            },
            coords={
                "a": self.new_x_coord_var(
                    attrs={"standard_name": "grid_longitude"}
                ),
                "b": self.new_y_coord_var(
                    attrs={"standard_name": "grid_latitude"}
                ),
                "lon": self.new_x_coord_var(),
                "lat": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_ROTATED_POLE,
            "rotated_latitude_longitude",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_and_standard_name_projected(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "a": self.new_x_coord_var(
                    attrs={"standard_name": "projection_x_coordinate"}
                ),
                "b": self.new_y_coord_var(
                    attrs={"standard_name": "projection_y_coordinate"}
                ),
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_UTM_33N,
            "transverse_mercator",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_and_axis(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "a": self.new_x_coord_var(attrs={"axis": "X"}),
                "b": self.new_y_coord_var(attrs={"axis": "Y"}),
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_UTM_33N,
            "transverse_mercator",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_and_common_dim_and_var_name(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "a": self.new_x_coord_var(),
                "b": self.new_y_coord_var(),
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_UTM_33N,
            "transverse_mercator",
            ("x", "y"),
            ("x", "y"),
            (20, 10)
        )

        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    dims=("time", "lat", "lon"),
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_WGS84)
            },
            coords={
                "a": self.new_x_coord_var(dim="lon"),
                "b": self.new_y_coord_var(dim="lat"),
                "lon": self.new_x_coord_var(dim="lon"),
                "lat": self.new_y_coord_var(dim="lat"),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_WGS84,
            "latitude_longitude",
            ("lon", "lat"),
            ("lon", "lat"),
            (20, 10)
        )

    def test_with_gm_var_and_common_dim_name(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    attrs={"grid_mapping": "crs"}
                ),
                "crs": self.new_crs_var(CRS_UTM_33N)
            },
            coords={
                "a": self.new_x_coord_var(),
                "b": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_UTM_33N,
            "transverse_mercator",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )

    def test_with_gm_var_but_without_spatial_var(self):
        ds = xr.Dataset(
            data_vars={
                "crs": self.new_crs_var(CRS_UTM_33N),
            },
            coords={
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "crs")
        self.assertIsNone(gmt)

    def test_without_gm_var_and_common_dim_and_var_name(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(),
            },
            coords={
                "a": self.new_x_coord_var(),
                "b": self.new_y_coord_var(),
                "x": self.new_x_coord_var(),
                "y": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_WGS84,
            "latitude_longitude",
            ("x", "y"),
            ("x", "y"),
            (20, 10)
        )

        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var(
                    dims=("time", "lat", "lon"),
                ),
            },
            coords={
                "a": self.new_x_coord_var(dim="lon"),
                "b": self.new_y_coord_var(dim="lat"),
                "lon": self.new_x_coord_var(dim="lon"),
                "lat": self.new_y_coord_var(dim="lat"),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_WGS84,
            "latitude_longitude",
            ("lon", "lat"),
            ("lon", "lat"),
            (20, 10)
        )

    def test_without_gm_var_and_common_dim_name(self):
        ds = xr.Dataset(
            data_vars={
                "sst": self.new_data_var()
            },
            coords={
                "a": self.new_x_coord_var(),
                "b": self.new_y_coord_var(),
            }
        )
        gmt = find_grid_mapping_for_data_var(ds, "sst")
        self.assert_grid_mapping_tuple_tuple_ok(
            gmt,
            CRS_WGS84,
            "latitude_longitude",
            ("a", "b"),
            ("x", "y"),
            (20, 10)
        )
