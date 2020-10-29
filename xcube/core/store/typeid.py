# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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

from typing import Any
from typing import Optional
from typing import Set
from typing import Union

from xcube.core.mldataset import MultiLevelDataset
from xcube.util.assertions import assert_given

import geopandas as gpd
import xarray as xr

class TypeId:
    """
    A type id denotes a type of data. It is used to group similar types of data and discern different types of data.
    It can be used by stores to state what types of data can be read from and/or written to them.

    A type id consists of a name and an arbitrary number of flags.
    Flags can be used to further refine a type id if needed.

    :param name: The name of the type of data
    :param flags: An arbitrary set of flags that further refine the type
    """

    def __init__(self, name: str, flags: Set[str] = None):
        assert_given(name, 'name')
        self._name = name
        self._flags = flags if flags is not None else set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def flags(self) -> Set[str]:
        return self._flags

    def __str__(self) -> str:
        if len(self.flags) == 0:
            return self.name
        flag_part = ','.join(sorted(self.flags))
        return f'{self.name}[{flag_part}]'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        try:
            other_type = self.normalize(other)
        except TypeError:
            return False
        if self.name != other_type.name:
            return False
        return len(self.flags.symmetric_difference(other_type.flags)) == 0

    def is_compatible(self, other: Union[str, "TypeId"]) -> bool:
        """
        Tests whether another type id is compatible with this type id.
        This is a weaker relationship than *equals()*.
        Two type ids are considered compatible when they have the same name and the other type id has all the flags
        that this type id has. The other type id may also have additional flags.

        :param other: Another type id, as string or *TypeId*.
        :return: Whether the other type id is compatible with this type id
        """
        other_type = self.normalize(other)
        if self.name != '*' and self.name != other_type.name:
            return False
        return len(self.flags.difference(other_type.flags)) == 0

    def __hash__(self) -> int:
        return hash(self.name) + 16 * hash(frozenset(self.flags))

    @classmethod
    def normalize(cls, type_id: Union[str, "TypeId"]) -> "TypeId":
        if isinstance(type_id, TypeId):
            return type_id
        if isinstance(type_id, str):
            return cls.parse(type_id)
        raise TypeError('type_id must be of type "str" or "TypeId"')

    @classmethod
    def parse(cls, type_id: str) -> "TypeId":
        if '[' not in type_id:
            return TypeId(type_id)
        if not type_id.endswith(']'):
            raise SyntaxError(f'"{type_id}" cannot be parsed: No end brackets found')
        name = type_id.split('[')[0]
        flags = type_id.split('[')[1].split(']')[0].split(',')
        return TypeId(name, flags=set(flags))

TYPE_ID_ANY = TypeId('*')
TYPE_ID_DATASET = TypeId('dataset')
TYPE_ID_CUBE = TypeId('dataset', flags={'cube'})
TYPE_ID_MULTI_LEVEL_DATASET = TypeId('mldataset')
TYPE_ID_GEO_DATA_FRAME = TypeId('geodataframe')

def get_type_id(data: Any) -> Optional[TypeId]:
    if isinstance(data, xr.Dataset):
        if 'time' in data.coords and 'lat' in data.coords and 'lon' in data.coords:
            return TYPE_ID_CUBE
        return TYPE_ID_DATASET
    elif isinstance(data, MultiLevelDataset):
        return TYPE_ID_MULTI_LEVEL_DATASET
    elif isinstance(data, gpd.GeoDataFrame):
        return TYPE_ID_GEO_DATA_FRAME
    return None
