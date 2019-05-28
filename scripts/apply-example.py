# This is an example for a Python script to by used with the "xcube apply" command.
# Usage:
#
#   $ xcube apply --vars refl_1,refl_4 --params a=2.5,b=0.5 output.zarr scripts/apply-example.py input.zarr
#


# TODO: support params() return a tuple (name, type, range, default) for each parameter,
# so "xcube apply" can validate passed parameters.
#
def params():
    return (("a", float, [0, 10], 0.5),
            ("b", float, [0, 10], 0.5))


def apply(variable_a, variable_b, a=0.5, b=0.5):
    print(variable_a.shape)
    print(variable_b.shape)
    return a * variable_a + b * variable_b


def init(*cubes, **params):
    if len(cubes) != 1:
        raise ValueError("Only one cube expected")

    cube = cubes[0]
    var_names = list(cube.data_vars)
    if len(var_names) < 2:
        raise ValueError("Cube must have two variables")

    illegal_param_names = set(params.keys()).difference({"a", "b"})
    if illegal_param_names:
        raise ValueError(f"Illegal parameter(s): {illegal_param_names}")

    var_name_1 = var_names[0]
    var_name_2 = var_names[1]
    print(var_name_1, var_name_2)
    return [cube[var_name_1], cube[var_name_2]], params
