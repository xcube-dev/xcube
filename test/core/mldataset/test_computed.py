import os
import unittest
import zipfile

from xcube.core.dsio import rimraf
from xcube.core.mldataset import BaseMultiLevelDataset
from xcube.core.mldataset import ComputedMultiLevelDataset
from .helpers import get_test_dataset


def get_input_ml_dataset_getter():
    my_ml_dataset = BaseMultiLevelDataset(get_test_dataset())

    def input_ml_dataset_getter(dataset_id):
        if dataset_id == "my_ml_dataset":
            return my_ml_dataset
        raise RuntimeError(f"unexpected dataset_id={dataset_id!r}")

    return input_ml_dataset_getter


def get_script_dir():
    script_dir = os.path.join(os.path.dirname(__file__), "test-code")
    if os.path.exists(script_dir):
        rimraf(script_dir)
    os.mkdir(script_dir)
    return script_dir


class ComputedMultiLevelDatasetTest(unittest.TestCase):
    def test_it(self):
        ml_ds2 = ComputedMultiLevelDataset(
            os.path.join(os.path.dirname(__file__),
                         "..", "..", "webapi", "res", "script.py"),
            "compute_dataset",
            ["my_ml_dataset"],
            get_input_ml_dataset_getter(),
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

    def test_import_from_dir(self):
        script_dir = get_script_dir()

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

        try:
            computed_ml_ds = ComputedMultiLevelDataset(
                f"{script_dir}/module_1.py",
                "compute_dataset",
                ["my_ml_dataset"],
                get_input_ml_dataset_getter(),
                ds_id="ml_ds2"
            )
            self.assert_computed_ml_dataset_ok(computed_ml_ds)
        finally:
            rimraf(script_dir)

    def test_import_from_zip(self):
        script_dir = get_script_dir()

        with zipfile.ZipFile(f"{script_dir}/modules.zip", "w") as zf:
            with zf.open(f"module_1.py", "w") as fp:
                fp.write(
                    b"import module_2 as m2\n"
                    b"\n"
                    b"def compute_dataset(ds):\n"
                    b"    return m2.process_dataset(ds)\n"
                )
            with zf.open(f"module_2.py", "w") as fp:
                fp.write(
                    b"\n"
                    b"def process_dataset(ds):\n"
                    b"    return ds.copy()\n"
                )

        try:
            computed_ml_ds = ComputedMultiLevelDataset(
                f"{script_dir}/modules.zip",
                "module_1:compute_dataset",
                ["my_ml_dataset"],
                get_input_ml_dataset_getter(),
                ds_id="ml_ds2"
            )
            self.assert_computed_ml_dataset_ok(computed_ml_ds)
        finally:
            rimraf(script_dir)

    def assert_computed_ml_dataset_ok(
            self,
            computed_ml_ds: ComputedMultiLevelDataset
    ):
        self.assertEqual(3, computed_ml_ds.num_levels)
        self.assertEqual(1, computed_ml_ds.num_inputs)
        self.assertEqual(computed_ml_ds.grid_mapping,
                         computed_ml_ds.get_input_dataset(0).grid_mapping)

        # assert output is same as input
        base_dataset = computed_ml_ds.get_dataset(0)
        self.assertEqual({'lon', 'lat',
                          'lat_bnds', 'lon_bnds',
                          'time'}, set(base_dataset.coords))
        self.assertEqual({'noise'}, set(base_dataset.data_vars))
        # assert we can compute without exception being raised:
        for i in range(computed_ml_ds.num_levels):
            computed_ml_ds.get_dataset(i).compute()
