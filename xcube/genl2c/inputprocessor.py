from abc import ABCMeta

import xarray as xr

from xcube.io import DatasetIO


class InputProcessor(DatasetIO, metaclass=ABCMeta):
    """
    Read and process inputs for the genl2c tool.
    """

    @property
    def modes(self):
        return {'r'}

    def write(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()

    def append(self, dataset: xr.Dataset, output_path: str, **kwargs):
        raise NotImplementedError()

    @classmethod
    def pre_reproject(cls, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        return dataset

    @classmethod
    def post_reproject(cls, dataset: xr.Dataset) -> xr.Dataset:
        """ Do any pre-processing before reprojection. """
        return dataset
