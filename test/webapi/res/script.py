import numpy as np
import xarray as xr

from xcube.core.mldataset import IdentityMultiLevelDataset
from xcube.core.mldataset import MultiLevelDataset


def compute_dataset(ds, period='1W', incl_stdev=False):
    if incl_stdev:
        resample_obj = ds.resample(time=period)
        ds_mean = resample_obj.mean(dim='time')
        ds_std = resample_obj.std(dim='time').rename(
            name_dict={name: f"{name}_stdev" for name in ds.data_vars})
        ds_merged = xr.merge([ds_mean, ds_std])
        ds_merged.attrs.update(ds.attrs)
        return ds_merged
    else:
        return ds.resample(time=period).mean(dim='time')


def compute_variables(ds, factor_chl, factor_tsm):
    chl_tsm_sum = factor_chl * ds.conc_chl + factor_tsm * ds.conc_tsm
    chl_tsm_sum.attrs.update(
        dict(units='-',
             long_name='Weighted sum of CHL nd TSM concentrations',
             description='Nonsense variable, for demo purpose only')
    )

    chl_category = _categorize_chl(ds.conc_chl)
    chl_category.attrs.update(
        dict(units='-',
             long_name='Categorized CHL',
             description='0: 0<=CHL<3, 1: 3<=CHL<4, 2: CHL>4 mg/m^3')
    )

    return xr.Dataset(
        dict(chl_tsm_sum=chl_tsm_sum,
             chl_category=chl_category)
    )


def _categorize_chl(chl):
    return xr.where(chl >= 4., 2,
                    xr.where(chl >= 3.0, 1,
                             xr.where(chl >= 0.0, 0,
                                      np.nan)))


class CopyMultiLevelDataset(IdentityMultiLevelDataset):
    """Example for a custom MultiLevelDataset class."""


def broken_ml_dataset_factory_1():
    """Example for a custom, broken MultiLevelDataset class."""
    return None


def broken_ml_dataset_factory_2(ml_dataset: MultiLevelDataset):
    """Example for a custom, broken MultiLevelDataset class."""
    return xr.Dataset()
