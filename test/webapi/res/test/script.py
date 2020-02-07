import xarray as xr


def compute_dataset(ds, period='1W', incl_stdev=False):
    if incl_stdev:
        resample_obj = ds.resample(time=period)
        ds_mean = resample_obj.mean(dim='time')
        ds_std = resample_obj.std(dim='time').rename(name_dict={name: f"{name}_stdev" for name in ds.data_vars})
        ds_merged = xr.merge([ds_mean, ds_std])
        ds_merged.attrs.update(ds.attrs)
        return ds_merged
    else:
        return ds.resample(time=period).mean(dim='time')


def compute_variables(ds, factor_chl, factor_tsm):
    chl_tsm_sum = factor_chl * ds.conc_chl + factor_tsm * ds.conc_tsm
    chl_tsm_sum.attrs.update(dict(units='-',
                                  long_name='Weighted sum of CHL nd TSM conc.',
                                  description='Nonsense variable, for demo purpose only'))
    return xr.Dataset(dict(chl_tsm_sum=chl_tsm_sum))
