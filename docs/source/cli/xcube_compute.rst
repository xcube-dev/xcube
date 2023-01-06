=================
``xcube compute``
=================

Synopsis
========

Compute a cube variable from other cube variables using a user-provided Python function.

::

    $ xcube compute --help

::

    Usage: xcube compute [OPTIONS] SCRIPT [CUBE]...

      Compute a cube from one or more other cubes.

      The command computes a cube variable from other cube variables in CUBEs
      using a user-provided Python function in SCRIPT.

      The SCRIPT must define a function named "compute":

          def compute(*input_vars: numpy.ndarray,
                      input_params: Mapping[str, Any] = None,
                      dim_coords: Mapping[str, np.ndarray] = None,
                      dim_ranges: Mapping[str, Tuple[int, int]] = None) \
                      -> numpy.ndarray:
              # Compute new numpy array from inputs
              # output_array = ...
              return output_array

      where input_vars are numpy arrays (chunks) in the order given by VARIABLES
      or given by the variable names returned by an optional "initialize" function
      that my be defined in SCRIPT too, see below. input_params is a mapping of
      parameter names to values according to PARAMS or the ones returned by the
      aforesaid "initialize" function. dim_coords is a mapping from dimension name
      to coordinate labels for the current chunk to be computed. dim_ranges is a
      mapping from dimension name to index ranges into coordinate arrays of the
      cube.

      The SCRIPT may define a function named "initialize":

          def initialize(input_cubes: Sequence[xr.Dataset],
                         input_var_names: Sequence[str],
                         input_params: Mapping[str, Any]) \
                         -> Tuple[Sequence[str], Mapping[str, Any]]:
              # Compute new variable names and/or new parameters
              # new_input_var_names = ...
              # new_input_params = ...
              return new_input_var_names, new_input_params

      where input_cubes are the respective CUBEs, input_var_names the respective
      VARIABLES, and input_params are the respective PARAMS. The "initialize"
      function can be used to validate the data cubes, extract the desired
      variables in desired order and to provide some extra processing parameters
      passed to the "compute" function.

      Note that if no input variable names are specified, no variables are passed
      to the "compute" function.

      The SCRIPT may also define a function named "finalize":

          def finalize(output_cube: xr.Dataset,
                       input_params: Mapping[str, Any]) \
                       -> Optional[xr.Dataset]:
              # Optionally modify output_cube and return it or return None
              return output_cube

      If defined, the "finalize" function will be called before the command writes
      the new cube and then exists. The functions may perform a cleaning up or
      perform side effects such as write the cube to some sink. If the functions
      returns None, the CLI will *not* write any cube data.

    Options:
      --variables, --vars VARIABLES  Comma-separated list of variable names.
      -p, --params PARAMS            Parameters passed as 'input_params' dict to
                                     compute() and init() functions in SCRIPT.
      -o, --output OUTPUT            Output path. Defaults to 'out.zarr'
      -f, --format FORMAT            Output format.
      -N, --name NAME                Output variable's name.
      -D, --dtype DTYPE              Output variable's data type.
      --help                         Show this message and exit.


Example
=======

::

    $ xcube compute s3-olci-cube.zarr ./algoithms/s3-olci-ndvi.py


with ``./algoithms/s3-olci-ndvi.py`` being:

::

    # TODO

Python API
==========

The related Python API function is :py:func:`xcube.core.compute.compute_cube`.

