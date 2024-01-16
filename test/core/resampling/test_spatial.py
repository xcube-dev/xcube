import unittest

import numpy as np
import pyproj

from xcube.core.gridmapping import CRS_WGS84
from xcube.core.gridmapping import GridMapping
from xcube.core.gridmapping.regular import RegularGridMapping
from xcube.core.new import new_cube
from xcube.core.resampling import resample_in_space

nan = np.nan


class ResampleInSpaceTest(unittest.TestCase):

    def test_affine_transform_dataset(self):
        source_ds = new_cube(variables={"CHL": 10., "TSM": 8.5})
        source_gm = GridMapping.from_dataset(source_ds)
        target_gm = GridMapping.regular(size=(8, 4),
                                        xy_min=(0, 0),
                                        xy_res=2,
                                        crs=CRS_WGS84)

        target_ds = resample_in_space(source_ds, source_gm, target_gm,
                                      encode_cf=True,
                                      gm_name="crs")

        self.assertIn("crs", target_ds)
        self.assertEqual(
            target_gm.crs,
            pyproj.CRS.from_cf(target_ds.crs.attrs)
        )

        for var_name in ("CHL", "TSM"):
            self.assertIn(var_name, target_ds)
            self.assertEqual(
                "crs",
                target_ds[var_name].attrs.get("grid_mapping")
            )

        actual_gm = GridMapping.from_dataset(target_ds)
        self.assertEqual(RegularGridMapping, type(target_gm))
        self.assertEqual(RegularGridMapping, type(actual_gm))
        self.assertEqual(actual_gm.crs, target_gm.crs)
        self.assertEqual(actual_gm.xy_res, target_gm.xy_res)
        self.assertEqual(actual_gm.xy_bbox, target_gm.xy_bbox)
        self.assertEqual(actual_gm.xy_dim_names, target_gm.xy_dim_names)

    # noinspection PyMethodMayBeStatic
    def test_rectify_and_downscale_dataset(self):
        # TODO: write more tests
        pass

    # noinspection PyMethodMayBeStatic
    def test_rectify_and_upscale_dataset(self):
        # TODO: write more tests
        pass
