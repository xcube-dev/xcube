from typing import Sequence, Mapping, Union, Dict, Any, Callable

from xcube.util.undefined import UNDEFINED

ParamName = str
ParamValue = Union[bool, int, float, str]
ParamValue = Union[ParamValue, Sequence[ParamValue], Mapping[Union[int, str], ParamValue]]
ParamValues = Mapping[ParamName, ParamValue]


class ParamDescriptor:
    def __init__(self,
                 name: str,
                 default: Any = UNDEFINED,
                 dtype: Union[type, str] = None,
                 description: str = None,
                 value_set: Sequence[ParamValue] = None,
                 validate_value: Callable = None,
                 to_json_value: Callable = None,
                 from_json_value: Callable = None, ):
        # TODO: validate params
        self.name = name
        self.default = default
        self.dtype = dtype
        self.description = description
        self.value_set = value_set
        self._validate_value = validate_value
        self._to_json_value = to_json_value
        self._from_json_value = from_json_value

    @property
    def required(self) -> bool:
        return self.default is UNDEFINED

    def validate_value(self, value: Any):
        if self._validate_value is not None:
            self._validate_value(value)

    def to_json_value(self, value: Any) -> Any:
        """Convert *value* from JSON value into Python object."""
        return self._to_json_value(value) if self._to_json_value is not None else value

    def from_json_value(self, value: Any) -> Any:
        """Convert *value* from Python object into JSON value."""
        return self._from_json_value(value) if self._from_json_value is not None else value


class ParamDescriptorSet:
    def __init__(self, param_descriptors: Sequence[ParamDescriptor]):
        self._param_descriptors = list(param_descriptors)

    @property
    def param_descriptors(self) -> Sequence[ParamDescriptor]:
        return list(self._param_descriptors)

    def from_json_values(self,
                         param_values: ParamValues = None,
                         exception_type=ValueError) -> Dict[str, Any]:
        """
        Parse and validate JSON-dict *param_values* and return a dictionary
        with values according to this parameter set.
        """

        parsed_param_values = dict()

        for param_descriptor in self._param_descriptors:
            if param_descriptor.name in param_values:
                param_value = param_descriptor.from_json_value(param_values[param_descriptor.name])
            elif not param_descriptor.required:
                param_value = param_descriptor.default
            else:
                raise exception_type(f'Missing required parameter "{param_descriptor.name}"')
            try:
                param_descriptor.validate_value(param_value)
            except ValueError as e:
                raise exception_type(str(e)) from e
            parsed_param_values[param_descriptor.name] = param_value

        param_set = {param_descriptor.name for param_descriptor in self._param_descriptors}
        for param_name in param_values.keys():
            if param_name not in param_set:
                raise exception_type(f'Unknown parameter "{param_name}"')

        return parsed_param_values
