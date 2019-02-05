import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from test.sampledata import create_highroc_dataset
from xcube.api.gen.snap.vectorize import vectorize_wavebands


class VectorizeTest(unittest.TestCase):
    def test_vectorize_spectra(self):
        dataset = create_highroc_dataset()
        self.assertEqual(36, len(dataset.data_vars))

        vectorized_dataset = vectorize_wavebands(dataset)

        self.assertEqual(36 - 2 * 16 + 2, len(vectorized_dataset.data_vars))
        self.assertIn('band', vectorized_dataset.dims)
        self.assertIn('band', vectorized_dataset.coords)
        band_var = vectorized_dataset.coords['band']
        assert_array_almost_equal(np.array([400., 412.5, 442.5, 490., 510., 560., 620., 665.,
                                            673.75, 681.25, 708.75, 753.75, 778.75, 865., 885., 940.]),
                                  band_var.values)
        self.assertEqual((16,), band_var.shape)
        self.assertEqual('nm', band_var.attrs.get('units'))

        self.assertIn('rtoa', vectorized_dataset)
        rtoa_var = vectorized_dataset['rtoa']
        self.assertEqual((16, 3, 4), rtoa_var.shape)
        self.assertEqual(('band', 'y', 'x'), rtoa_var.dims)
        self.assertEqual('1', rtoa_var.attrs.get('units'))
        self.assertEqual('Top-of-atmosphere reflectance',
                         rtoa_var.attrs.get('long_name'))

        self.assertIn('rrs', vectorized_dataset)
        rrs_var = vectorized_dataset['rrs']
        self.assertEqual((16, 3, 4), rrs_var.shape)
        self.assertEqual(('band', 'y', 'x'), rrs_var.dims)
        self.assertEqual('sr^-1', rrs_var.attrs.get('units'))
        self.assertEqual('Atmospherically corrected angular dependent remote sensing reflectances',
                         rrs_var.attrs.get('long_name'))

    def test_vectorize_nothing(self):
        dataset = create_highroc_dataset(no_spectra=True)
        vectorized_dataset = vectorize_wavebands(dataset)
        self.assertIs(dataset, vectorized_dataset)

