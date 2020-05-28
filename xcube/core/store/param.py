from typing import Sequence, Mapping, Union

ParamName = str
ParamValue = Union[bool, int, float, str]
ParamValue = Union[ParamValue, Sequence[ParamValue], Mapping[Union[int, str], ParamValue]]


class ParamDescriptor:
    def __init__(self,
                 name: ParamName,
                 default: ParamValue,
                 description: str = None,
                 value_set: Sequence[ParamValue] = None):
        # TODO: validate params
        self.name = name
        self.default = default
        self.description = description
        self.value_set = value_set
