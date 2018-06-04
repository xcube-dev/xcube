from abc import ABCMeta

import xarray as xr

from .mask import mask_dataset
from ..inputprocessor import InputProcessor


class SnapInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Input processor for SNAP L2 NetCDF inputs.
    """

    def __init__(self, expr_pattern=None):
        self.expr_pattern = expr_pattern

    def read(self, input_file: str, **kwargs) -> xr.Dataset:
        """ Read SNAP L2 NetCDF inputs. """
        return xr.open_dataset(input_file, decode_cf=True, decode_coords=True, decode_times=False)

    @classmethod
    def pre_reproject(cls, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        masked_dataset, _ = mask_dataset(dataset,
                                         expr_pattern='({expr}) AND !quality_flags.land',
                                         errors='raise')
        return masked_dataset


# noinspection PyAbstractClass
class SnapOlciHighrocL2InputProcessor(SnapInputProcessor):
    """
    Input processor for SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs.
    """

    def __init__(self):
        super().__init__(expr_pattern='({expr}) AND !quality_flags.land')

    @property
    def name(self) -> str:
        return 'snap-olci-highroc-l2'

    @property
    def description(self) -> str:
        return 'SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs'

    @property
    def ext(self) -> str:
        return 'nc'
