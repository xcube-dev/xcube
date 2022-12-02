import os
import unittest

from xcube.core.dsio import rimraf
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import ComputedMultiLevelDataset
from .helpers import get_test_dataset


class ComputedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ds = get_test_dataset()

        ml_ds1 = BaseMultiLevelDataset(ds)

        def input_ml_dataset_getter(ds_id):
            if ds_id == "ml_ds1":
                return ml_ds1
            self.fail(f"unexpected ds_id={ds_id!r}")

        ml_ds2 = ComputedMultiLevelDataset(
            os.path.join(os.path.dirname(__file__),
                         "..", "..", "webapi", "res", "script.py"),
            "compute_dataset",
            ["ml_ds1"],
            input_ml_dataset_getter,
            input_parameters=dict(period='1W'),
            ds_id="ml_ds2"
        )
        self.assertEqual(3, ml_ds2.num_levels)

        ds0 = ml_ds2.get_dataset(0)
        self.assertEqual({'time': 3, 'lat': 720, 'lon': 1440, 'bnds': 2},
                         ds0.dims)

        ds1 = ml_ds2.get_dataset(1)
        self.assertEqual({'time': 3, 'lat': 360, 'lon': 720}, ds1.dims)

        ds2 = ml_ds2.get_dataset(2)
        self.assertEqual({'time': 3, 'lat': 180, 'lon': 360}, ds2.dims)

        self.assertEqual([ds0, ds1, ds2], ml_ds2.datasets)

        ml_ds1.close()
        ml_ds2.close()

    def test_import(self):
        script_dir = os.path.join(os.path.dirname(__file__), "test-code")

        if os.path.exists(script_dir):
            rimraf(script_dir)

        os.mkdir(script_dir)

        with open(f"{script_dir}/module_1.py", "w") as fp:
            fp.write(
                "import module_2 as m2\n"
                "\n"
                "def compute_dataset(ds):\n"
                "    return m2.process_dataset(ds)\n"
            )

        with open(f"{script_dir}/module_2.py", "w") as fp:
            fp.write(
                "\n"
                "def process_dataset(ds):\n"
                "    return ds.copy()\n"
            )

        ds = get_test_dataset()

        ml_ds1 = BaseMultiLevelDataset(ds)

        def input_ml_dataset_getter(ds_id):
            if ds_id == "ml_ds1":
                return ml_ds1
            self.fail(f"unexpected ds_id={ds_id!r}")

        try:
            ml_ds2 = ComputedMultiLevelDataset(
                f"{script_dir}/module_1.py",
                "compute_dataset",
                ["ml_ds1"],
                input_ml_dataset_getter,
                input_parameters=dict(),
                ds_id="ml_ds2"
            )
            ml_ds2.get_dataset(0).compute()
        finally:
            rimraf(script_dir)
