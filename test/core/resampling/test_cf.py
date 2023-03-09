import unittest

from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube
from xcube.core.resampling import encode_grid_mapping


class EncodeGridMappingTest(unittest.TestCase):
    def test_geographic_and_force_encoding(self):
        ds = new_cube(variables={"chl": 0.6, "tsm": 0.8})
        self.assertEqual({"chl", "tsm"}, set(ds.data_vars))

        gm = GridMapping.from_dataset(ds)

        gm_ds = encode_grid_mapping(ds, gm,
                                    gm_name="spatial_ref",
                                    force=True)

        self.assertEqual({"chl", "tsm", "spatial_ref"}, set(gm_ds.data_vars))
        self.assertEqual("spatial_ref", gm_ds.chl.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", gm_ds.tsm.attrs.get("grid_mapping"))

    def test_geographic_and_do_not_force_encoding(self):
        ds = new_cube(variables={"chl": 0.6, "tsm": 0.8})
        self.assertEqual({"chl", "tsm"}, set(ds.data_vars))

        gm = GridMapping.from_dataset(ds)

        gm_ds = encode_grid_mapping(ds, gm,
                                    gm_name="spatial_ref",
                                    force=False)

        self.assertEqual({"chl", "tsm"}, set(gm_ds.data_vars))
        self.assertEqual(None, gm_ds.chl.attrs.get("grid_mapping"))
        self.assertEqual(None, gm_ds.tsm.attrs.get("grid_mapping"))

    def test_replace_gm_name(self):
        ds = new_cube(x_start=10000, y_start=10000,
                      x_res=10, y_res=10,
                      width=100, height=100,
                      x_name="x",
                      y_name="y",
                      crs='EPSG:3857',
                      crs_name="spatial_ref",
                      variables={"chl": 0.6, "tsm": 0.8})
        self.assertEqual({"chl", "tsm", "spatial_ref"}, set(ds.data_vars))

        gm = GridMapping.regular(size=(100, 100),
                                 xy_min=(10000., 10000.),
                                 xy_res=(10., 10.),
                                 crs='EPSG:3857')

        # Use former gm name "spatial_ref"
        gm_ds = encode_grid_mapping(ds, gm)

        self.assertEqual({"chl", "tsm", "spatial_ref"}, set(gm_ds.data_vars))
        self.assertEqual("spatial_ref", gm_ds.chl.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", gm_ds.tsm.attrs.get("grid_mapping"))

        # Use new gm name "srs"
        gm_ds = encode_grid_mapping(ds, gm,
                                    gm_name="srs")

        self.assertEqual({"chl", "tsm", "srs"}, set(gm_ds.data_vars))
        self.assertEqual("srs", gm_ds.chl.attrs.get("grid_mapping"))
        self.assertEqual("srs", gm_ds.tsm.attrs.get("grid_mapping"))

        # Use default gm name "crs"
        gm_ds = encode_grid_mapping(ds.drop_vars("spatial_ref"), gm,
                                    force=True)

        self.assertEqual({"chl", "tsm", "crs"}, set(gm_ds.data_vars))
        self.assertEqual("crs", gm_ds.chl.attrs.get("grid_mapping"))
        self.assertEqual("crs", gm_ds.tsm.attrs.get("grid_mapping"))

