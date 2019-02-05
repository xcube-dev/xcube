import numpy as np
import pandas as pd
import xarray as xr


def new_test_dataset(time, height=180, **indexers):
    """
    Create a test dataset with dimensions ("time", "lat", "lon") and data variables given by *indexers*.

    :param time: Single date/time string or sequence of date-time strings.
    :param height: Size of the latitude dimension.
    :param indexers: Variable name to value mapping.
           Value may be a scalar or a vector of same length as *time*.
    :return: test dataset
    """
    # TODO (forman): get rid of this code here, utilise xcube.api.new_cube() instead
    time = [time] if isinstance(time, str) else time
    width = height * 2
    num_times = len(time)
    res = 180 / height
    shape = (1, height, width)
    data_vars = dict()
    for name, value in indexers.items():
        try:
            values = list(value)
        except TypeError:
            values = [value] * num_times
        if len(values) != num_times:
            raise ValueError()
        data_vars[name] = (['time', 'lat', 'lon'],
                           np.concatenate(tuple(np.full(shape, values[i]) for i in range(num_times))))
    return xr.Dataset(data_vars,
                      coords=dict(time=(['time'], pd.to_datetime(time)),
                                  lat=(['lat'], np.linspace(-90 + res, +90 - res, height)),
                                  lon=(['lon'], np.linspace(-180 + res, +180 - res, width))))


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
                units="degrees_east",
            )),
            lat=(('y', 'x'), lat, dict(
                long_name="latitude",
                units="degrees_north",
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
                     [8, 1, 1, 1]], dtype=np.uint32)
    return xr.DataArray(data, dims=('y', 'x'), name='c2rcc_flags', attrs=dict(
        long_name="C2RCC quality flags",
        _Unsigned="true",
        flag_meanings="F1 F2 F3 F4",
        flag_masks=np.array([1, 2, 4, 8], np.int32),
        flag_coding_name="c2rcc_flags",
        flag_descriptions="D1 D2 D3 D4",
    ))


def create_cmems_sst_flag_var():
    sea = 1
    land = 2
    lake = 4
    ice = 8
    data = np.array([[[sea + ice, land + ice, lake + ice, lake],
                      [sea + ice, sea, land, land],
                      [sea, sea, sea, land]]], dtype=np.float32)
    return xr.DataArray(data, dims=('time', 'lat', 'lon'), name='mask', attrs=dict(
        long_name="land sea ice lake bit mask",
        flag_masks="0b, 1b, 2b, 3b",
        flag_meanings="sea land lake ice",
        valid_min=0,
        valid_max=12,
    ))
