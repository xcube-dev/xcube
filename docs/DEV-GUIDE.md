# Xcube Developer Guide

Version 0.1, draft

*IMPORTANT NOTE: Any changes to this doc must be reviewed by dev-team through pull requests.* 

## Table of Contents

- [Versioning](#versioning)
- [Coding Style](#coding-style)
- [Main Packages](#main-packages)
  - [Package `xcube.cli`](#package-xcubecli)
  - [Package `xcube.api`](#package-xcubeapi)
  - [Package `xcube.webapi`](#package-xcubewebapi)
- [Development Process](#development-process)

## Versioning

We adhere to [PEP-440](https://www.python.org/dev/peps/pep-0440/).

The current software version is in `xcube/version.py`.

While developing a version, we append version suffix `.dev<N>`.
Before the release, we remove the suffix.

## Coding Style

We try adhering to [PEP-8](https://www.python.org/dev/peps/pep-0008/).

## Main Packages

* `xcube.cli` - Here live CLI commands that are required by someone. 
  CLI command implementations should be lightweight. 
  Move implementation code either into `api` or `util`.  
  CLI commands must be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing CLI commands. 
* `xcube.api` - Here live API functions that are required by someone or that exists because a CLI 
  command is implemented here. 
  API code must be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing API. 
* `xcube.webapi` - Here live Web API functions that are required by someone. 
  Web API command implementations should be lightweight.
  Move implementation code either into `api` or `util`.  
  Web API interface must be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing API.
* `xcube.util` - Mainly implementation helpers. 
  Comprises classes and functions that are used by `cli`, `api`, `webapi` 
  in order to maximize modularisation and testability but to minimize code duplication.  
  The code in here must not be dependent on any of `cli`, `api`, `webapi`.
  The code in here may change often and in any way as desired by code 
  implementing the `cli`, `api`, `webapi` packages.    

The following sections will guide you through extending or changing the main packages that form
xcube's public interface.

### Package `xcube.cli`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/api`); 
1. is reflected by the CLI's doc-strings and tools documentation (currently in `README.md`);
1. follows existing xcube CLI conventions;
1. follows PEP8 conventions;
1. is reflected in API and WebAPI, if desired;
1. is reflected in `CHANGES.md`.

#### Hints

Make sure your new CLI command is in line with the others commands regarding 
command name, option names, as well as metavar arguments names. 
The CLI command name shall ideally be a verb.

Avoid introducing new option arguments if similar options are already in use 
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

### Package `xcube.api`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/api`); 
1. is covered by API documentation;
1. follows existing xcube API conventions;
1. follows PEP8 conventions;
1. is reflected in xarray extension class `xcube.api.api.API`;
1. is reflected in CLI and WebAPI if desired;
1. is reflected in `CHANGES.md`.

#### Hints

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

If `import xcube.api` is used in client code, any `xarray.Dataset` object will have
an extra property `xcube` whose interface is defined by the class 
`xcube.api.XCubeAPI`. This class is an 
[xarray extension](http://xarray.pydata.org/en/stable/internals.html#extending-xarray) that is 
used to reflect `xcube.api` functions and make it directly applicable to the `xarray.Dataset` object.

Therefore any xcube API shall be reflected in this extension class.


### Package `xcube.webapi`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/webapi`); 
1. is covered by Web API specification and documentation (currently in `webapi/res/openapi.yml`);
1. follows existing xcube Web API conventions;
1. follows PEP8 conventions;
1. is reflected in CLI and API, if desired;
1. is reflected in `CHANGES.md`.

### Hints

* The Web API is defined in `webapi.app` which defines mapping from resource URLs to handlers
* All handlers are implemented in `webapi.handlers`. Handler code just delegates to dedicated controllers.
* All controllers are implemented in `webapi.controllers.*`. They might further delegate into `api.*`

## Development Process

1. Make sure there is an issue ticket for your code change work item
1. Select issue, priorities are as follows 
   1. "urgent" and ("important" and "bug")
   1. "urgent" and ("important" or "bug")
   1. "urgent"
   1. "important" and "bug"
   1. "important" or "bug"
   1. others
1. Make sure issue is assigned to you, if unclear agree with team first.
1. Add issue label "in progress".
1. Create development branch named "developer-issue#-title".
1. Develop, having in mind the checklists and implementation hints above.
   1. In your first commit, refer the issue so it will appear as link in the issue history
   1. Develop, test, and push to the remote branch as desired. 
   1. In your last commit, utilize checklists above. 
      (You can include the line "closes #<issue>" in your commit message to auto-close the issue 
      once the PR is merged.)
1. Create PR if build servers succeed on your branch. If not, fix issue first.  
   For the PR assign the team for review, agree who is to merge. 
   Also reviewers must have checklist in mind! 
1. Merge PR after all reviewers are accepted your change. Otherwise go back. 
1. Remove issue label "in progress".
1. Delete the development branch "developer-issue#-title".
1. If the PR is only partly solving an issue:
    1. Do not include the line "closes #<issue>" in your last commit message. 
    1. Add "relates to issue#" in PR.  
    1. Make sure to check the corresponding boxes after the PR is merged.
    1. Remove issue label "in progress". 
    1. Leave issue open. 
