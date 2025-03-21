#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.

import xarray as xr


def compute_dataset(ds, period="1W", incl_stdev=False):
    if incl_stdev:
        resample_obj = ds.resample(time=period)
        ds_mean = resample_obj.mean(dim="time")
        ds_std = resample_obj.std(dim="time").rename(
            name_dict={name: f"{name}_stdev" for name in ds.data_vars}
        )
        ds_merged = xr.merge([ds_mean, ds_std])
        ds_merged.attrs.update(ds.attrs)
        return ds_merged
    else:
        return ds.resample(time=period).mean(dim="time")
