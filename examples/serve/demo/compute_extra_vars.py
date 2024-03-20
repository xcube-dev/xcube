import numpy as np
import xarray as xr


def compute_variables(ds, factor_chl, factor_tsm):
    chl_tsm_sum = factor_chl * ds.conc_chl + factor_tsm * ds.conc_tsm
    chl_tsm_sum.attrs.update(
        dict(
            units="-",
            long_name="Weighted sum of CHL nd TSM concentrations",
            description="Nonsense variable, for demo purpose only",
        )
    )

    chl_category = _categorize_chl(ds.conc_chl)
    chl_category.attrs.update(
        color_value_min=0.0,
        color_value_max=2.0,
        color_bar_name="spring",
        units="-",
        long_name="Chlorophyll bloom risk",
        description="Chlorophyll bloom risk in three categories: "
        "0: CHL < 3, "
        "1: 3 <= CHL < 4, "
        "2: CHL >= 4 mg/m^3",
    )

    return xr.Dataset(dict(chl_tsm_sum=chl_tsm_sum, chl_category=chl_category))


def _categorize_chl(chl):
    return xr.where(
        chl >= 4.0, 2, xr.where(chl >= 3.0, 1, xr.where(chl >= 0.0, 0, np.nan))
    )
