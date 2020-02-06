import xarray as xr


def compute_variables(ds, factor_chl, factor_tsm):
    chl_tsm_sum = factor_chl * ds.conc_chl + factor_tsm * ds.conc_tsm
    chl_tsm_sum.attrs.update(dict(units='-',
                                  long_name='Weighted sum of CHL nd TSM conc.',
                                  description='Nonsense variable, for demo purpose only'))
    return xr.Dataset(dict(chl_tsm_sum=chl_tsm_sum))
