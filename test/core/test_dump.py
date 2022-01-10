import unittest

from test.sampledata import new_test_dataset
from xcube.core.dump import dump_dataset


class DumpDatasetTest(unittest.TestCase):
    def test_dump_dataset(self):
        dataset = new_test_dataset(["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
                                   precipitation=0.4, temperature=275.2)
        for var in dataset.variables.values():
            var.encoding.update({"_FillValue": 999.0})

        print(dataset.dims)

        text = dump_dataset(dataset)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (time: 5, lat: 180, lon: 360)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertNotIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertNotIn("Encoding for data variable 'temperature':\n", text)
        self.assertNotIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, show_var_encoding=True)
        self.assertIn("<xarray.Dataset>", text)
        self.assertIn("Dimensions:        (time: 5, lat: 180, lon: 360)\n", text)
        self.assertIn("Coordinates:\n", text)
        self.assertIn("  * lon            (lon) float64 ", text)
        self.assertIn("Data variables:\n", text)
        self.assertIn("    precipitation  (time, lat, lon) float64 ", text)
        self.assertIn("Encoding for coordinate variable 'lat':\n", text)
        self.assertIn("Encoding for data variable 'temperature':\n", text)
        self.assertIn("    _FillValue:  999.0\n", text)

        text = dump_dataset(dataset, ["precipitation"])
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertNotIn("Encoding:\n", text)
        self.assertNotIn("    _FillValue:  999.0", text)

        text = dump_dataset(dataset, ["precipitation"], show_var_encoding=True)
        self.assertIn("<xarray.DataArray 'precipitation' (time: 5, lat: 180, lon: 360)>\n", text)
        self.assertIn("Encoding:\n", text)
        self.assertIn("    _FillValue:  999.0", text)
