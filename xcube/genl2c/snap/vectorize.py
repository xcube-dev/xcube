from typing import Callable

import numpy as np
import xarray as xr

BandCoordVarFactory = Callable[[str, np.ndarray], xr.DataArray]


def new_band_coord_var(band_dim_name: str, band_values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(band_values,
                        name=band_dim_name,
                        dims=band_dim_name,
                        attrs=dict(units='nm',
                                   standard_name='sensor_band_central_radiation_wavelength',
                                   long_name='band_wavelength'))


def vectorize_wavebands(dataset: xr.Dataset,
                        band_coord_var_factory: BandCoordVarFactory = new_band_coord_var) -> xr.Dataset:
    bands_dict = {}

    for var_name in dataset.data_vars:
        variable = dataset.data_vars[var_name]
        wavelength = variable.attrs.get('wavelength')
        wavelength = float(wavelength if wavelength is not None else -1.0)
        if wavelength > 0.0:
            i = len(var_name) - 1
            while i >= 0 and (var_name[i].isdigit() or var_name[i] == '_'):
                i -= 1
            spectrum_name = var_name[0:i + 1]
            if spectrum_name in bands_dict:
                band = bands_dict[spectrum_name]
            else:
                band = []
                bands_dict[spectrum_name] = band
            band.append((wavelength, variable))

    if not bands_dict:
        return dataset

    band_list = []
    band_dim_index = 0
    band_dim_count = 0
    for spectrum_name, band in bands_dict.items():
        # Sort each band_values by wavelength
        band_values, spectrum_variables = zip(*sorted(band, key=lambda e: e[0]))
        band_values = np.array(band_values, dtype=np.float64)
        if len(band_list) > 0:
            # Make sure we use the same band_dim_index for equal band_values
            band_dim_index_1 = None
            for band_values_2, band_dim_index_2, _, _ in band_list:
                if band_values.shape == band_values_2.shape and np.allclose(band_values, band_values_2):
                    band_dim_index_1 = band_dim_index_2
                    break
            if band_dim_index_1 is not None:
                band_dim_index = band_dim_index_1
            else:
                band_dim_count += 1
                band_dim_index = band_dim_count
        band_list.append((band_values, band_dim_index, spectrum_name, spectrum_variables))

    # Remove waveband variables from dataset
    dropped_var_names = {spectrum_variable.name
                         for _, _, _, spectrum_variables in band_list
                         for spectrum_variable in spectrum_variables}
    dataset = dataset.drop(dropped_var_names)

    # And replace by new vectorized waveband / spectrum variables
    for band_values, band_dim_index, spectrum_name, spectrum_variables in band_list:
        band_dim_name = 'band' if band_dim_count == 0 else 'band' + str(band_dim_index + 1)
        band_coord_var = band_coord_var_factory(band_dim_name, band_values)
        spectrum_variable = xr.concat(spectrum_variables, dim=band_dim_name)
        time_coord_size = spectrum_variable.sizes.get('time', 0)
        if time_coord_size == 1 and spectrum_variable.dims[0] != 'time':
            spectrum_variable = spectrum_variable.squeeze('time')
            spectrum_variable = spectrum_variable.expand_dims('time')
        dataset[spectrum_name] = spectrum_variable
        dataset = dataset.assign_coords(**{band_dim_name: band_coord_var})

    return dataset
