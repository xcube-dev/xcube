"""
This script is used to generate the test dataset in `test/gen/snap/inputdata`.
"""

import glob
import os

import xarray as xr

NC_PATH = 'C:\\Users\\Norman\\OneDrive - Brockmann Consult GmbH\\EOData\\HIGHROC\\O_L2_0001_SNS_*_v1.0.nc'

for file in glob.glob(NC_PATH):
    ds = xr.open_dataset(file)
    v = set(ds.data_vars.keys())
    v2 = set()
    v2.update({x for x in v if x.startswith('atmos')})
    v2.update({x for x in v if x.startswith('unc_')})
    v2.update({x for x in v if x.startswith('tur_')})
    v2.update({x for x in v if x.startswith('spm_')})
    v2.update({x for x in v if x.startswith('rhown_')})
    v2.update({x for x in v if x.startswith('iop_')})
    v2.update({x for x in v if x.startswith('meta')})
    v2.update({x for x in v if x.startswith('zsd')})
    v2.update({x for x in v if x.startswith('total')})
    v2.update({x for x in v if x.startswith('horiz')})
    v2.update({x for x in v if x.startswith('chl')})
    v2.update({x for x in v if x.endswith('_mask')})
    v2.update({x for x in v if x.endswith('_flag')})
    ds2 = ds.drop(v2)
    ds3 = ds2.isel(y=slice(0, 100), x=slice(0, 100))
    ds4 = ds3.isel(tp_x=slice(0, 102), tp_y=slice(0, 102))
    ds4.to_netcdf(os.path.basename(file))
