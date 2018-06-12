import numpy as np
import xarray as xr


def create_highroc_dataset(no_spectra=False):
    """
    Simulates a HIGHROC OLCI L2 product in NetCDF 4 format
    """
    lon = np.array([[8, 9.3, 10.6, 11.9],
                    [8, 9.2, 10.4, 11.6],
                    [8, 9.1, 10.2, 11.3]], dtype=np.float32)
    lat = np.array([[56, 56.1, 56.2, 56.3],
                    [55, 55.2, 55.4, 55.6],
                    [54, 54.3, 54.6, 54.9]], dtype=np.float32)

    if not no_spectra:
        wavelengths = [(1, 400.0), (2, 412.5), (3, 442.5), (4, 490.0), (5, 510.0),
                       (6, 560.0), (7, 620.0), (8, 665.0), (9, 673.75), (10, 681.25),
                       (11, 708.75), (12, 753.75), (16, 778.75), (17, 865.0), (18, 885.0), (21, 940.0)]
        rtoa_desc = "Top-of-atmosphere reflectance"
        rrs_desc = "Atmospherically corrected angular dependent remote sensing reflectances"
        rtoa_vars = {f'rtoa_{i}': create_waveband(i, wl, '1', rtoa_desc) for i, wl in wavelengths}
        rrs_vars = {f'rrs_{i}': create_waveband(i, wl, 'sr^-1', rrs_desc) for i, wl in wavelengths}
    else:
        rtoa_vars = {}
        rrs_vars = {}

    return xr.Dataset(
        data_vars=dict(
            conc_chl=create_conc_chl(),
            c2rcc_flags=create_c2rcc_flag_var(),
            lon=(('y', 'x'), lon, dict(
                long_name="longitude",
                units="degrees east",
            )),
            lat=(('y', 'x'), lat, dict(
                long_name="latitude",
                units="degrees north",
            )),
            **rtoa_vars,
            **rrs_vars,
        ),
        attrs=dict(start_date='14-APR-2017 10:27:50.183264',
                   stop_date='14-APR-2017 10:31:42.736226')
    )


def create_waveband(index, wavelength, units, long_name=None):
    data = np.array([[7, 11, np.nan, 5],
                     [5, 10, 2, 21],
                     [16, 6, 20, 17]], dtype=np.float32)
    return (('y', 'x'), data, dict(
        long_name=long_name,
        units=units,
        spectral_band_index=index,
        wavelength=wavelength,
        bandwidth=15.0,
        valid_pixel_expression="c2rcc_flags.F1",
        _FillValue=np.nan,
    ))


def create_conc_chl():
    data = np.array([[7, 11, np.nan, 5],
                     [5, 10, 2, 21],
                     [16, 6, 20, 17]], dtype=np.float32)
    return (('y', 'x'), data, dict(
        long_name="Chlorophylll concentration",
        units="mg m^-3",
        _FillValue=np.nan,
        valid_pixel_expression="c2rcc_flags.F1",
    ))


def create_c2rcc_flag_var():
    data = np.array([[1, 1, 1, 1],
                     [1, 4, 1, 2],
                     [8, 1, 1, 1]], dtype=np.uint8)
    return xr.DataArray(data, dims=('y', 'x'), name='c2rcc_flags', attrs=dict(
        long_name="C2RCC quality flags",
        _Unsigned="true",
        flag_meanings="F1 F2 F3 F4",
        flag_masks=np.array([1, 2, 4, 8], np.int32),
        flag_coding_name="c2rcc_flags",
        flag_descriptions="D1 D2 D3 D4",
    ))
