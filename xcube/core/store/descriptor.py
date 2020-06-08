from typing import Tuple, Sequence, Mapping, Union, Optional, Dict, Any


# TODO: IMPORTANT: support/describe also multi-resolution cubes (*.levels)
# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.schema.CubeSchema class
#   xcube.webapi.context.DatasetDescriptor type
#   responses of xcube.webapi.controllers.catalogue
# TODO: rename to CubeDescriptor? Currently called after xr.Dataset class
# TODO: write tests
# TODO: document me
# TODO: validate params
class DatasetDescriptor:

    def __init__(self,
                 dataset_id: str,
                 dims: Mapping[str, int] = None,
                 data_vars: Sequence['VariableDescriptor'] = None,
                 attrs: Mapping[str, any] = None,
                 spatial_crs: str = None,
                 spatial_coverage: Tuple[float, float, float, float] = None,
                 spatial_resolution: Union[float, Tuple[float, float]] = None,
                 temporal_coverage: Tuple[Optional[str], Optional[str]] = None,
                 temporal_resolution: str = None):
        self.id = dataset_id
        self.dims = dict(dims) if dims else None
        self.data_vars = list(data_vars) if data_vars else None
        self.attrs = dict(attrs) if attrs else None
        self.spatial_crs = spatial_crs
        self.spatial_coverage = spatial_coverage
        self.spatial_resolution = spatial_resolution
        self.temporal_coverage = temporal_coverage
        self.temporal_resolution = temporal_resolution

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        # TODO: implement me


# TODO: write tests
# TODO: document me
# TODO: validate params
class VariableDescriptor:

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 attrs: Mapping[str, any] = None):
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.attrs = dict(attrs or {})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        # TODO: implement me
