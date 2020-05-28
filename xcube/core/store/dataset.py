from typing import Tuple, Sequence, Mapping, Union, Optional


class VariableDescriptor:

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 attrs: Mapping[str, any] = None):
        # TODO: validate params
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.attrs = dict(attrs or {})


class DatasetDescriptor:
    def __init__(self,
                 id: str,
                 dims: Mapping[str, int] = None,
                 data_vars: Sequence[VariableDescriptor] = None,
                 attrs: Mapping[str, any] = None,
                 spatial_crs: str = None,
                 spatial_coverage: Tuple[float, float, float, float] = None,
                 spatial_resolution: Union[float, Tuple[float, float]] = None,
                 temporal_coverage: Tuple[Optional[str], Optional[str]] = None,
                 temporal_resolution: str = None):
        # TODO: validate params
        self.id = id
        self.dims = dict(dims)
        self.data_vars = list(data_vars)
        self.attrs = dict(attrs or {})
        self.spatial_crs = spatial_crs
        self.spatial_coverage = spatial_coverage
        self.spatial_resolution = spatial_resolution
        self.temporal_coverage = temporal_coverage
        self.temporal_resolution = temporal_resolution
