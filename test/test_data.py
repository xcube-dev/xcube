import numpy as np
import xarray as xr


def create_highroc_dataset():
    """
    Simulates a HIGHROC OLCI L2 product in NetCDF 4 format
    """
    lon = np.array([[8, 9.3, 10.6, 11.9],
                    [8, 9.2, 10.4, 11.6],
                    [8, 9.1, 10.2, 11.3]], dtype=np.float32)
    lat = np.array([[56, 56.1, 56.2, 56.3],
                    [55, 55.2, 55.4, 55.6],
                    [54, 54.3, 54.6, 54.9]], dtype=np.float32)
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

        ),
        attrs=dict(start_date='14-APR-2017 10:27:50.183264',
                   stop_date='14-APR-2017 10:31:42.736226')
    )


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
