# noinspection PyUnusedLocal
def compute(variable_a, variable_b, input_params=None, **kwargs):
    a = input_params.get('a', 0.5)
    b = input_params.get('b', 0.5)
    return a * variable_a + b * variable_b


def initialize(input_cubes, input_var_names, input_params):
    if len(input_cubes) != 1:
        raise ValueError("Expected a single input cube")

    if input_var_names and len(input_var_names) != 2:
        raise ValueError("Two variables expected")

    input_cube = input_cubes[0]

    if not input_var_names:
        if 'precipitation' not in input_cube:
            raise ValueError("Cube must have 'precipitation'")
        if 'soil_moisture' not in input_cube:
            raise ValueError("Cube must have 'soil_moisture'")
        input_var_names = ['precipitation', 'soil_moisture']

    illegal_param_names = set(input_params.keys()).difference({"a", "b"})
    if illegal_param_names:
        raise ValueError(f"Illegal parameter(s): {illegal_param_names}")

    return input_var_names, dict(a=input_params.get('a', 0.2), b=input_params.get('b', 0.3))


def finalize(output_cube):
    output_cube.output.attrs.update(units='mg/m^3')
    output_cube.attrs.update(comment='I has a bucket')
    return output_cube
