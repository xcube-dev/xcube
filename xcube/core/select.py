from typing import Collection

import xarray as xr


def select_vars(dataset: xr.Dataset, var_names: Collection[str] = None) -> xr.Dataset:
    """
    Select data variable from given *dataset* and create new dataset.

    :param dataset: The dataset from which to select variables.
    :param var_names: The names of data variables to select.
    :return: A new dataset. It is empty, if *var_names* is empty. It is *dataset*, if *var_names* is None.
    """
    if var_names is None:
        return dataset
    dropped_variables = set(dataset.data_vars.keys()).difference(var_names)
    if not dropped_variables:
        return dataset
    return dataset.drop_vars(dropped_variables)
