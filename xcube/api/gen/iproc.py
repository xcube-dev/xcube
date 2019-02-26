# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Optional, Collection

import xarray as xr

from xcube.util.objreg import get_obj_registry
from xcube.util.reproject import reproject_xy_to_wgs84


class ReprojectionInfo:

    def __init__(self,
                 xy_var_names: Tuple[str, str],
                 xy_tp_var_names: Tuple[str, str] = None,
                 xy_crs: str = None,
                 xy_gcp_step: Union[int, Tuple[int, int]] = None,
                 xy_tp_gcp_step: Union[int, Tuple[int, int]] = None):
        """
        Characterize input datasets so we can reproject.

        :param xy_var_names: Name of variable providing the spatial x- and y-coordinates,
               e.g. ('lon', 'lat')
        :param xy_tp_var_names: Name of tie-point variable providing the spatial y- and y-coordinates,
               e.g. ('TP_longitude', 'TP_latitude')
        :param xy_crs: Spatial reference system, e.g. 'EPSG:4326'
        :param xy_gcp_step: Step size for collecting ground control points from spatial
               coordinate arrays given by **xy_tp_var_names**.
        :param xy_tp_gcp_step: Step size for collecting ground control points from spatial
               coordinate arrays given by **xy_var_names**.
        """
        self.xy_var_names = xy_var_names
        self.xy_tp_var_names = xy_tp_var_names
        self.xy_crs = xy_crs
        self.xy_gcp_step = xy_gcp_step
        self.xy_tp_gcp_step = xy_tp_gcp_step


class InputProcessor(metaclass=ABCMeta):
    """
    Read and process inputs for the gen tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The name of this input processor
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        :return: The description of this input processor
        """

    def configure(self, **parameters):
        """
        Configure this input processor.
        :param parameters: The configuration parameters.
        :return: The description of this input processor
        """
        if parameters:
            raise TypeError(f"got an unexpected input processor parameters {parameters!r}")

    @property
    @abstractmethod
    def input_reader(self) -> str:
        """
        :return: The input reader for this input processor.
        """

    @property
    def input_reader_params(self) -> dict:
        """
        :return: The input reader parameters for this input processor.
        """
        return dict()

    @abstractmethod
    def get_time_range(self, dataset: xr.Dataset) -> Optional[Tuple[float, float]]:
        """
        Return a tuple of two floats representing start/stop time (which may be same) in days since 1970.
        :param dataset: The dataset.
        :return: The time-range tuple of the dataset or None.
        """

    def get_extra_vars(self, dataset: xr.Dataset) -> Optional[Collection[str]]:
        """
        Get a set of variables names that are required for the pre-processing and reprojection steps and should
        therefore not be dropped. However, the reprojection or post-processing steps may later remove them.

        Returns ``None`` by default.

        :param dataset: The dataset.
        :return: Collection of names of variables to be prevented from being dropping.
        """
        return None

    def pre_process(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Do any pre-processing before reprojection. All variables of the output must be 2D arrays.
        For example, perform dataset validation, masking, and/or filtering using provided configuration parameters.
        The default implementation returns the unchanged *dataset*.
        :param dataset: The dataset.
        :return: The pre-processed dataset or the original one.
        """
        return dataset

    @abstractmethod
    def reproject(self,
                  dataset: xr.Dataset,
                  dst_size: Tuple[int, int],
                  dst_region: Tuple[float, float, float, float],
                  dst_resampling: str,
                  include_non_spatial_vars=False) -> xr.Dataset:
        """
        Perform reprojection. The output must be a 2D array.
        :param dataset: The input dataset.
        :param dst_size: The output size in pixels as tuple (width ,height).
        :param dst_region: The output region in coordinates of the target CRS. A tuple (x_min, y_min, x_max, y_max).
        :param dst_resampling: The resampling method.
        :param include_non_spatial_vars: Whether to include non-spatial variables in the output.
        :return: The reprojected output dataset or the original one.
        """
        raise NotImplementedError()

    def post_process(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Do any post-processing after reprojection. The input is a 3D array with dimensions ("time", "lat", "lon").
        For example, generate new "wavelength" dimension for variables whose name follow a certain pattern.
        The default implementation returns the unchanged *dataset*.
        :param dataset: The dataset.
        :return: The post-processed dataset or the original one.
        """
        return dataset


class XYInputProcessor(InputProcessor, metaclass=ABCMeta):
    """
    Read and process inputs for the gen tool.
    """

    @abstractmethod
    def get_reprojection_info(self, dataset: xr.Dataset) -> Optional[ReprojectionInfo]:
        """
        Information about special fields in input datasets used for reprojection.
        :param dataset: The dataset.
        :return: The reprojection information of the dataset or None.
        """

    def get_extra_vars(self, dataset: xr.Dataset) -> Optional[Collection[str]]:
        """
        Return the names of variables containing tie-points.
        They should not be removed, as they are required for the reprojection.
        """
        reprojection_info = self.get_reprojection_info(dataset)
        if reprojection_info is None:
            return dataset
        extra_vars = set()
        if reprojection_info.xy_var_names:
            extra_vars.update(reprojection_info.xy_var_names)
        if reprojection_info.xy_tp_var_names:
            extra_vars.update(reprojection_info.xy_tp_var_names)
        return extra_vars

    def reproject(self,
                  dataset: xr.Dataset,
                  dst_size: Tuple[int, int],
                  dst_region: Tuple[float, float, float, float],
                  dst_resampling: str,
                  include_non_spatial_vars=False) -> xr.Dataset:
        """
        Perform reprojection using tie-points / ground control points.
        """
        reprojection_info = self.get_reprojection_info(dataset)
        if reprojection_info is None:
            return dataset
        return reproject_xy_to_wgs84(dataset,
                                     src_xy_var_names=reprojection_info.xy_var_names,
                                     src_xy_tp_var_names=reprojection_info.xy_tp_var_names,
                                     src_xy_crs=reprojection_info.xy_crs,
                                     src_xy_gcp_step=reprojection_info.xy_gcp_step or 1,
                                     src_xy_tp_gcp_step=reprojection_info.xy_tp_gcp_step or 1,
                                     dst_size=dst_size,
                                     dst_region=dst_region,
                                     dst_resampling=dst_resampling,
                                     include_non_spatial_vars=include_non_spatial_vars)


def register_input_processor(input_processor: InputProcessor):
    get_obj_registry().put(input_processor.name, input_processor, type=InputProcessor)


def get_input_processor(name: str):
    return get_obj_registry().get(name, type=InputProcessor)
