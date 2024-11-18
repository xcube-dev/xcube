# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import List

import click

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"

DEFAULT_OUTPUT_PATH = "out.zarr"


@click.command(name="compute")
@click.argument("script")
@click.argument("cube", nargs=-1)
@click.option(
    "--variables",
    "--vars",
    "input_var_names",
    metavar="VARIABLES",
    help="Comma-separated list of variable names.",
)
@click.option(
    "--params",
    "-p",
    "input_params",
    metavar="PARAMS",
    help="Parameters passed as 'input_params' dict to compute() and init() functions in SCRIPT.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    metavar="OUTPUT",
    default=DEFAULT_OUTPUT_PATH,
    help=f"Output path. Defaults to {DEFAULT_OUTPUT_PATH!r}",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    metavar="FORMAT",
    default="zarr",
    type=click.Choice(["zarr", "nc"]),
    help="Output format.",
)
@click.option(
    "--name",
    "-N",
    "output_var_name",
    metavar="NAME",
    default="output",
    help="Output variable's name.",
)
@click.option(
    "--dtype",
    "-D",
    "output_var_dtype",
    metavar="DTYPE",
    default="float64",
    type=click.Choice(
        ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]
    ),
    help="Output variable's data type.",
)
def compute(
    script: str,
    cube: list[str],
    input_var_names: str,
    input_params: str,
    output_path: str,
    output_format: str,
    output_var_name: str,
    output_var_dtype: str,
):
    """Compute a cube from one or more other cubes.

    The command computes a cube variable from other cube variables in CUBEs
    using a user-provided Python function in SCRIPT.

    The SCRIPT must define a function named "compute":

    \b
        def compute(*input_vars: numpy.ndarray,
                    input_params: Mapping[str, Any] = None,
                    dim_coords: Mapping[str, np.ndarray] = None,
                    dim_ranges: Mapping[str, Tuple[int, int]] = None) \\
                    -> numpy.ndarray:
            # Compute new numpy array from inputs
            # output_array = ...
            return output_array

    where input_vars are numpy arrays (chunks) in the order given by VARIABLES or given by the variable names returned
    by an optional "initialize" function that my be defined in SCRIPT too, see below. input_params is a mapping of
    parameter names to values according to PARAMS or the ones returned by the aforesaid "initialize" function.
    dim_coords is a mapping from dimension name to coordinate labels for the current chunk to be computed.
    dim_ranges is a mapping from dimension name to index ranges into coordinate arrays of the cube.

    The SCRIPT may define a function named "initialize":

    \b
        def initialize(input_cubes: Sequence[xr.Dataset],
                       input_var_names: Sequence[str],
                       input_params: Mapping[str, Any]) \\
                       -> Tuple[Sequence[str], Mapping[str, Any]]:
            # Compute new variable names and/or new parameters
            # new_input_var_names = ...
            # new_input_params = ...
            return new_input_var_names, new_input_params

    where input_cubes are the respective CUBEs, input_var_names the respective VARIABLES, and input_params
    are the respective PARAMS. The "initialize" function can be used to validate the data cubes, extract
    the desired variables in desired order and to provide some extra processing parameters passed to the
    "compute" function.

    Note that if no input variable names are specified, no variables are passed to the "compute" function.

    The SCRIPT may also define a function named "finalize":

    \b
        def finalize(output_cube: xr.Dataset,
                     input_params: Mapping[str, Any]) \\
                     -> Optional[xr.Dataset]:
            # Optionally modify output_cube and return it or return None
            return output_cube

    If defined, the "finalize" function will be called before the command writes the
    new cube and then exists. The functions may perform a cleaning up or perform side effects such
    as write the cube to some sink. If the functions returns None, the CLI will *not* write
    any cube data.

    """
    from xcube.cli.common import parse_cli_kwargs
    from xcube.core.compute import compute_cube
    from xcube.core.dsio import open_cube
    from xcube.core.dsio import guess_dataset_format, find_dataset_io

    input_paths = cube

    compute_function_name = "compute"
    initialize_function_name = "initialize"
    finalize_function_name = "finalize"

    with open(script) as fp:
        code = fp.read()

    locals_dict = dict()
    exec(code, globals(), locals_dict)

    input_var_names = (
        list(map(lambda s: s.strip(), input_var_names.split(",")))
        if input_var_names
        else None
    )

    compute_function = _get_function(
        locals_dict, compute_function_name, script, force=True
    )
    initialize_function = _get_function(
        locals_dict, initialize_function_name, script, force=False
    )
    finalize_function = _get_function(
        locals_dict, finalize_function_name, script, force=False
    )

    input_params = parse_cli_kwargs(input_params, "PARAMS")

    input_cubes = []
    for input_path in input_paths:
        input_cubes.append(open_cube(input_path=input_path))

    if initialize_function:
        input_var_names, input_params = initialize_function(
            input_cubes, input_var_names, input_params
        )

    output_cube = compute_cube(
        compute_function,
        *input_cubes,
        input_var_names=input_var_names,
        input_params=input_params,
        output_var_name=output_var_name,
        output_var_dtype=output_var_dtype,
    )

    if finalize_function:
        output_cube = finalize_function(output_cube)

    if output_cube is not None:
        output_format = output_format or guess_dataset_format(output_path)
        dataset_io = find_dataset_io(output_format, {"w"})
        dataset_io.write(output_cube, output_path)


def _get_function(object, function_name, source, force=False):
    function = object.get(function_name)
    if function is None:
        if force:
            raise click.ClickException(
                f"missing function {function_name!r} in {source}"
            )
        else:
            return None
    if not callable(function):
        raise click.ClickException(f"{function_name!r} in {source} is not a callable")
    return function
