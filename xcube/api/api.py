from contextlib import contextmanager
from typing import Dict, List, Mapping, Any, Union, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .chunk import chunk_dataset
from .dump import dump_dataset
from .new import new_cube
from .extract import get_cube_values_for_points, get_cube_point_indexes, get_cube_values_for_indexes, \
    get_dataset_indexes, INDEX_NAME_PATTERN
from .readwrite import read_cube, open_cube, write_cube
from .select import select_vars
from .verify import verify_cube


@xr.register_dataset_accessor('xcube')
class XCubeAPI:
    """
    The XCube API.

    :param dataset: An xarray dataset instance, that should conform to xcube's data cube specification.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @classmethod
    def new(cls,
            title="Test Cube",
            width=360,
            height=180,
            spatial_res=1.0,
            lon_start=-180.0,
            lat_start=-90.0,
            time_periods=5,
            time_freq="D",
            time_start='2010-01-01T00:00:00',
            drop_bounds=False,
            variables=None) -> xr.Dataset:
        """
        Create a new empty cube. Useful for testing.

        :param title: A title.
        :param width: Horizontal number of grid cells
        :param height: Vertical number of grid cells
        :param spatial_res: Spatial resolution in degrees
        :param lon_start: Minimum longitude value
        :param lat_start: Minimum latitude value
        :param time_periods: Number of time steps
        :param time_freq: Duration of each time step
        :param time_start: First time value
        :param drop_bounds: If True, coordinate bounds variables are not created.
        :param variables: Dictionary of data variables to be added.
        :return: A cube instance
        """
        return new_cube(title=title,
                        width=width,
                        height=height,
                        spatial_res=spatial_res,
                        lon_start=lon_start,
                        lat_start=lat_start,
                        time_periods=time_periods,
                        time_freq=time_freq,
                        time_start=time_start,
                        drop_bounds=drop_bounds,
                        variables=variables)

    @contextmanager
    @classmethod
    def open(cls, input_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
        """
        The ``read`` method as context manager that auto-closes the data cube read.

        :param input_path: input path
        :param format_name: format, e.g. "zarr" or "netcdf4"
        :param kwargs: format-specific keyword arguments
        :return: dataset object
        """
        yield open_cube(input_path, format_name=format_name, is_cube=True, **kwargs)

    @classmethod
    def read(cls, input_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
        """
        Read a data cube from *input_path*.
        If *format* is not provided it will be guessed from *input_path*.

        :param input_path: input path
        :param format_name: format, e.g. "zarr" or "netcdf4"
        :param kwargs: format-specific keyword arguments
        :return: dataset object
        """
        return read_cube(input_path, format_name=format_name, **kwargs)

    def write(self, output_path: str, format_name: str = None, **kwargs) -> xr.Dataset:
        """
        Write this cube to *output_path*.
        If *format* is not provided it will be guessed from *output_path*.

        :param output_path: output path
        :param format_name: format, e.g. "zarr" or "netcdf4"
        :param kwargs: format-specific keyword arguments
        :return: the input dataset
        """
        return write_cube(self._dataset, output_path, format_name=format_name, **kwargs)

    def values_for_points(self,
                          points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                          var_names: Sequence[str] = None,
                          index_name_pattern: str = INDEX_NAME_PATTERN,
                          include_indexes: bool = False,
                          method: str = "nearest",
                          cube_asserted: bool = False):
        """
        Extract values from cube variables at given coordinates in *points*.

        :param points: Dictionary that maps dimension name to coordinate arrays.
        :param var_names: An optional list of names of data variables in *cube* whose values shall be extracted.
        :param index_name_pattern: A naming pattern for the computed indexes columns.
               Must include "{name}" which will be replaced by the dimension name.
        :param include_indexes: Weather to include computed indexes in return value.
        :param method: "nearest" or "linear".
        :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
        :return: A new data frame whose columns are values from *cube* variables at given *points*.
        """
        return get_cube_values_for_points(self._dataset,
                                          points,
                                          var_names=var_names,
                                          index_name_pattern=index_name_pattern,
                                          include_indexes=include_indexes,
                                          method=method,
                                          cube_asserted=cube_asserted)

    def values_for_indexes(self,
                           indexes: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                           var_names: Sequence[str] = None,
                           index_name_pattern: str = INDEX_NAME_PATTERN,
                           method: str = "nearest",
                           cube_asserted: bool = False) -> xr.Dataset:
        """
        Get values from this cube at given *indexes*.

        :param indexes: A mapping from column names to index and fraction arrays for all cube dimensions.
        :param var_names: An optional list of names of data variables in *cube* whose values shall be extracted.
        :param index_name_pattern: A naming pattern for the computed indexes columns.
               Must include "{name}" which will be replaced by the dimension name.
        :param method: "nearest" or "linear".
        :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
        :return: A new data frame whose columns are values from *cube* variables at given *indexes*.
        """
        return get_cube_values_for_indexes(self._dataset,
                                           indexes,
                                           var_names=var_names,
                                           index_name_pattern=index_name_pattern,
                                           method=method,
                                           cube_asserted=cube_asserted)

    def point_indexes(self,
                      points: Union[xr.Dataset, pd.DataFrame, Mapping[str, Any]],
                      dim_name_mapping: Mapping[str, str] = None,
                      index_name_pattern: str = INDEX_NAME_PATTERN,
                      dtype=np.float64,
                      cube_asserted: bool = False):
        """
        Get indexes of given coordinates in *points* into this cube.

        :param points: A mapping from column names to column data arrays, which must all have the same length.
        :param dim_name_mapping: A mapping from dimension names in *cube* to column names in *points*.
        :param index_name_pattern: A naming pattern for the computed indexes columns.
               Must include "{name}" which will be replaced by the dimension name.
        :param cube_asserted: If False, *cube* will be verified, otherwise it is expected to be a valid cube.
        :param dtype: Numpy data type for the indexes. If it is floating point type (default),
               then *indexes* contain fractions, which may be used for interpolation. If *dtype* is an integer
               type out-of-range coordinates are indicated by index -1, and NaN if it is is a floating point type.
        :return: A dataset containing the index columns.
        """
        return get_cube_point_indexes(self._dataset,
                                      points,
                                      dim_name_mapping=dim_name_mapping,
                                      index_name_pattern=index_name_pattern,
                                      dtype=dtype,
                                      cube_asserted=cube_asserted)

    def indexes(self,
                coord_var_name: str,
                coord_values: Union[xr.DataArray, np.ndarray],
                dtype=np.float64) -> Union[xr.DataArray, np.ndarray]:
        """
        Compute the indexes into a coordinate variable *coord_var_name* of this cube
        for the given coordinate values *coord_values*.

        The coordinate variable's labels must be monotonic increasing or decreasing,
        otherwise the result will be nonsense.

        For any value in *coord_values* that is out of the bounds of the coordinate variable's values,
        the index depends on the value of *dtype*. If *dtype* is an integer type, invalid indexes are
        encoded as -1 while for floating point types, NaN will be used.

        Returns the indexes as an array-like object of type *dtype*.

        :param coord_var_name: Name of a coordinate variable.
        :param coord_values: Array-like coordinate values.
        :param dtype: (Numpy) data type for the indexes. If it is floating point type (default),
               then *indexes* contain fractions, which may be used for interpolation. If *dtype* is an integer
               type out-of-range coordinates are indicated by index -1, and NaN if it is is a floating point type.
        :return: The indexes and their fractions as a tuple of numpy int64 and float64 arrays.
        """
        return get_dataset_indexes(self._dataset,
                                   coord_var_name=coord_var_name,
                                   coord_values=coord_values,
                                   dtype=dtype)

    def chunk(self, chunk_sizes: Dict[str, int] = None, format_name: str = None) -> xr.Dataset:
        """
        Chunk this dataset and update encodings for given format.

        :param chunk_sizes: mapping from dimension name to new chunk size
        :param format_name: format, e.g. "zarr" or "netcdf4"
        :return: the re-chunked dataset
        """
        return chunk_dataset(self._dataset,
                             chunk_sizes=chunk_sizes,
                             format_name=format_name)

    def dump(self,
             variable_names=None,
             show_var_encoding=False) -> str:
        """
        Dump this dataset or its variables into a text string.

        :param variable_names: names of variables to be dumped
        :param show_var_encoding: also dump variable encodings?
        :return: the dataset dump
        """
        return dump_dataset(self._dataset,
                            variable_names=variable_names,
                            show_var_encoding=show_var_encoding)

    def verify(self) -> List[str]:
        """
        Verify that this dataset is a valid data cube.

        Returns a list of issues, which is empty if this dataset is a valid data cube.

        :return: List of issues or empty list.
        """
        return verify_cube(self._dataset)

    def select_vars(self, var_names: Sequence[str] = None):
        """
        Select data variable from given *dataset* and create new dataset.

        :param dataset: The dataset from which to select variables.
        :param var_names: The names of data variables to select.
        :return: A new dataset. It is empty, if *var_names* is empty. It is *dataset*, if *var_names* is None.
        """
        return select_vars(self._dataset, var_names)
