import xarray as xr


def vectorize_wavebands(dataset: xr.Dataset) -> xr.Dataset:

    spectra = {}

    for var_name in dataset.data_vars:
        var = dataset.data_vars[var_name]
        wavelength = var.attrs.get('wavelength')
        wavelength = float(wavelength if wavelength is not None else -1.0)
        if wavelength > 0.0:
            i = len(var_name) - 1
            while i >= 0 and (var_name[i].isdigit() or var_name[i] == '_'):
                i -= 1
            spectrum_name = var_name[0:i + 1]
            if spectrum_name in spectra:
                spectrum = spectra[spectrum_name]
            else:
                spectrum = []
                spectra[spectrum_name] = spectrum
            spectrum.append((wavelength, var))

    if not spectra:
        return dataset

    # Sort each spectrum by wavelength
    spectrum_list = []
    for spectrum_name, spectrum in spectra.items():
        sorted_spectrum = sorted(spectrum, key=lambda e: e[0])
        spectrum_list.append((spectrum_name, sorted_spectrum))
    # Sort list of spectra by length of spectrum
    sorted_spectrum_list = sorted(spectrum_list, key=lambda e: len(e[1]))

    # Remove old spectrum variables
    dropped_var_names = {wl[1].name for spectrum in spectra.values() for wl in spectrum}
    dataset = dataset.drop(dropped_var_names)

    # And replace by new vectorized spectrum variables
    for spectrum_name, spectrum in sorted_spectrum_list:
        sorted_spectrum = sorted(spectrum, key=lambda wl: wl[0])
        wavelengths, variables = zip(*sorted_spectrum)
        wavelengths = list(wavelengths)
        spectrum_var = xr.concat(variables, dim='band')
        spectrum_var = spectrum_var.assign_coords(band=wavelengths)
        band_var = spectrum_var.coords['band']
        band_var = band_var.assign_attrs(units='nm')
        spectrum_var.coords['band'] = band_var
        dataset[spectrum_name] = spectrum_var

    return dataset

