from typing import Tuple, Sequence, Mapping, Optional, Dict, Any

from xcube.util.assertions import assert_given
from xcube.util.ipython import register_json_formatter


# TODO: IMPORTANT: replace, reuse, or align with
#   xcube.core.schema.CubeSchema class
#   xcube.webapi.context.DatasetDescriptor type
#   responses of xcube.webapi.controllers.catalogue
# TODO: write tests
# TODO: document me
# TODO: validate params

class DataDescriptor:

    def __init__(self,
                 data_id: str,
                 type_id: str,
                 crs: str = None,
                 bbox: Tuple[float, float, float, float] = None,
                 spatial_res: float = None,
                 time_range: Tuple[Optional[str], Optional[str]] = None,
                 time_period: str = None):
        assert_given(data_id, 'data_id')
        assert_given(type_id, 'type_id')
        self.data_id = data_id
        self.type_id = type_id
        self.crs = crs
        self.bbox = tuple(bbox) if bbox else None
        self.spatial_res = spatial_res
        self.time_range = tuple(time_range) if time_range else None
        self.time_period = time_period

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict()
        _copy_none_null_props(self, d, ['data_id', 'type_id',
                                        'crs', 'bbox', 'spatial_res',
                                        'time_range', 'time_period'])
        return d


class DatasetDescriptor(DataDescriptor):

    def __init__(self,
                 data_id: str,
                 type_id='dataset',
                 dims: Mapping[str, int] = None,
                 data_vars: Sequence['VariableDescriptor'] = None,
                 attrs: Mapping[str, any] = None,
                 **kwargs):
        assert_given(data_id, 'data_id')
        super().__init__(data_id=data_id, type_id=type_id, **kwargs)
        self.dims = dict(dims) if dims else None
        self.data_vars = list(data_vars) if data_vars else None
        self.attrs = dict(attrs) if attrs else None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        if self.data_vars is not None:
            d['data_vars'] = [vd.to_dict() for vd in self.data_vars]
        _copy_none_null_props(self, d, ['dims', 'attrs'])
        return d


# TODO: write tests
# TODO: document me
class VariableDescriptor:

    def __init__(self,
                 name: str,
                 dtype: str,
                 dims: Sequence[str],
                 attrs: Mapping[str, any] = None):
        assert_given(name, 'name')
        assert_given(dtype, 'dtype')
        self.name = name
        self.dtype = dtype
        self.dims = tuple(dims)
        self.ndim = len(self.dims)
        self.attrs = dict(attrs) if attrs is None else None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = dict(name=self.name, dtype=self.dtype)
        _copy_none_null_props(self, d, ['dims', 'ndim', 'attrs'])
        return d


def _copy_none_null_props(obj: Any, d: Dict[str, Any], names: Sequence[str]):
    for name in names:
        value = getattr(obj, name)
        if value is not None:
            d[name] = value


class MultiLevelDatasetDescriptor(DatasetDescriptor):

    def __init__(self,
                 data_id: str,
                 num_levels: int,
                 **kwargs):
        assert_given(data_id, 'data_id')
        assert_given(num_levels, 'num_levels')
        super().__init__(data_id=data_id, type_id='mldataset', **kwargs)
        self.num_levels = num_levels

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'MultiLevelDatasetDescriptor':
        """Create new instance from a JSON-serializable dictionary"""
        # TODO: implement me

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary"""
        d = super().to_dict()
        d['num_levels'] = self.num_levels
        return d


register_json_formatter(DataDescriptor)
register_json_formatter(DatasetDescriptor)
register_json_formatter(VariableDescriptor)
register_json_formatter(MultiLevelDatasetDescriptor)
