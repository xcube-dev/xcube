# Xcube Developer Guide

Version 0.1, draft

*IMPORTANT NOTE: Any changes to this doc must be reviewed by dev-team through pull requests.* 

## Index

TODO

## Adding new functionality

Checklist:

1. Add API function, unit-tests and documentation
2. Add CLI function, unit-tests and documentation
3. Add xarray extension, unit-tests and documentation
4. Update tools documentation in README.md 
5. Update CHANGES.md

### New API

Create new module in `xcube.api` and add your functions.
For any functions added make sure naming is in line with other API.
Add clear doc-string to the new API. Use Sphinx RST format.

Decide if your API methods requires cubes as inputs, 
if so, name the primary dataset argument `cube` and add a 
keyword parameter `cube_asserted: bool = False`. 
Otherwise name the primary dataset argument `dataset`.

In the implementation, if not `cube_asserted`, 
we must assert the `cube` is a cube. 
Pass `True` to `cube_asserted` argument of other API called later on: 
    
    from .verify import assert_cube

    def frombosify_cube(cube: xr.Dataset, ..., cube_asserted: bool = False):  
        if not cube_asserted:
            assert_cube(cube)
        ...
        result = bibosify_cube(cube, ..., cube_asserted=True)
        ...

### New CLI

Make sure your new CLI command is in line with the others commands regarding 
command name, option names, as well as metavar arguments names. 
The CLI command name shall ideally be a verb.

Avoid introducing new option arguments if similar options already in use 
for existing commands.

In the following common arguments and options are listed.

Input argument:

    @click.argument('input', metavar='<input>')

Output argument:

    @click.argument('output', metavar='<output>')

Output (directory) option:

    @click.option('--output', '-o', metavar='<output>',
                  help='Output directory. If omitted, "<input>.levels" will be used.')

Output format:

    @click.option('--format', '-f', metavar='<format>', type=click.Choice(['zarr', 'netcdf']),
                  help="Format of the output. If not given, guessed from <output>.")

Output parameters:

    @click.option('--param', '-p', metavar='<param>', multiple=True,
                  help="Parameter specific for the output format. Multiple allowed.")

Variable names:

    @click.option('--var', '-v', metavar='<variable>', multiple=True,
                  help="Name of a variable. Multiple allowed.")


For parsing CLI inputs, use helper functions that are already in use.
In the CLI command implementation code, raise `click.ClickException(message)` 
with a clear `message` for users.

Extensively validate CLI inputs to avoid that API functions raise 
`ValueError`, `TypeError`, etc. Such errors and their message texts are
usually hard to understand by users. They are actually dedicated to 
to developers, not CLI users.

There is a global option `--traceback` flag that user can set to dump stack traces. 
You don't need to print stack traces from your code.  

### New xarray Extension

TODO

### Updating documentation

TODO

## Maintaining existing functionality

TODO

## Versioning

We adhere to [PEP-440](https://www.python.org/dev/peps/pep-0440/).

The current software version is in `xcube/version.py`.

While developing a version, we append version suffix `.dev<N>`.
Before the release, we remote it.

## Code style

We try adhering to [PEP-8](https://www.python.org/dev/peps/pep-0008/).

