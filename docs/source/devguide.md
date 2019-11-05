# xcube Developer Guide

Version 0.2, draft

*IMPORTANT NOTE: Any changes to this doc must be reviewed by dev-team 
through pull requests.* 

## Preface

> _Gedacht ist nicht gesagt._  
> _Gesagt ist nicht gehört._  
> _Gehört ist nicht verstanden._  
> _Verstanden ist nicht einverstanden._  
> _Einverstanden ist nicht umgesetzt._  
> _Umgesetzt ist nicht beibehalten._  
       
by Konrad Lorenz (translation is left to the reader)


## Table of Contents

- [Versioning](#versioning)
- [Coding Style](#coding-style)
- [Main Packages](#main-packages)
  - [Package `xcube.core`](#package-xcubecore)
  - [Package `xcube.cli`](#package-xcubecli)
  - [Package `xcube.webapi`](#package-xcubewebapi)
  - [Package `xcube.util`](#package-xcubeutil)
- [Development Process](#development-process)

## Versioning

We adhere to [PEP-440](https://www.python.org/dev/peps/pep-0440/).
Therefore, the xcube software version uses the format 
`<major>.<minor>.<micro>`  for released versions and 
`<major>.<minor>.<micro>.dev<n>` for versions in development. 

* `<major>` is increased for major enhancements. 
  CLI / API changes may introduce incompatibilities with former version.
* `<minor>` is increased for new features and and minor enhancements.
  CLI / API changes are backward compatible with former version.  
* `<micro>` is increased for bug fixes and micro enhancements.
  CLI / API changes are backward compatible with former version.  
* `<n>` is increased whenever the team (internally) deploys new builds
  of a development snapshot.

The current software version is in `xcube/version.py`.


## Coding Style

We try adhering to [PEP-8](https://www.python.org/dev/peps/pep-0008/).

## Main Packages

* `xcube.core` - Hosts core API functions. 
  Code in here should be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing core API. 
* `xcube.cli` - Hosts CLI commands. 
  CLI command implementations should be lightweight. 
  Move implementation code either into `core` or `util`.  
  CLI commands must be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing CLI 
  commands. 
* `xcube.webapi` - Hosts Web API functions. 
  Web API command implementations should be lightweight.
  Move implementation code either into `core` or `util`.  
  Web API interface must be maintained w.r.t. backward compatibility.
  Therefore think twice before adding new or change existing web API.
* `xcube.util` - Mainly implementation helpers. 
  Comprises classes and functions that are used by `cli`, `core`, 
  `webapi` in order to maximize modularisation and testability but to 
  minimize code duplication.  
  The code in here must not be dependent on any of `cli`, `core`,
  `webapi`. The code in here may change often and in any way as desired 
  by code implementing the `cli`, `core`, `webapi` packages.    

The following sections will guide you through extending or changing the
main packages that form xcube's public interface.

### Package `xcube.cli`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/cli`); 
1. is reflected by the CLI's doc-strings and tools documentation 
   (currently in `README.md`);
1. follows existing xcube CLI conventions;
1. follows PEP8 conventions;
1. is reflected in API and WebAPI, if desired;
1. is reflected in `CHANGES.md`.

#### Hints

Make sure your new CLI command is in line with the others commands 
regarding command name, option names, as well as metavar arguments 
names. The CLI command name shall ideally be a verb.

Avoid introducing new option arguments if similar options are already 
in use for existing commands.

In the following common arguments and options are listed.

Input argument:

    @click.argument('input')

If input argument is restricted to an xcube dataset:

    @click.argument('cube')


Output (directory) option:

    @click.option('--output', '-o', metavar='OUTPUT',
                  help='Output directory. If omitted, "INPUT.levels" will be used.')

Output format:

    @click.option('--format', '-f', metavar='FORMAT', type=click.Choice(['zarr', 'netcdf']),
                  help="Format of the output. If not given, guessed from OUTPUT.")

Output parameters:

    @click.option('--param', '-p', metavar='PARAM', multiple=True,
                  help="Parameter specific for the output format. Multiple allowed.")

Variable names:

    @click.option('--variable',--var', metavar='VARIABLE', multiple=True,
                  help="Name of a variable. Multiple allowed.")


For parsing CLI inputs, use helper functions that are already in use.
In the CLI command implementation code, raise 
`click.ClickException(message)` with a clear `message` for users.

Common xcube CLI options like `-f` for FORMAT should be lower case 
letters and specific xcube CLI options like `-S` for SIZE in `xcube gen`
are recommended to be uppercase letters. 

Extensively validate CLI inputs to avoid that API functions raise 
`ValueError`, `TypeError`, etc. Such errors and their message texts are
usually hard to understand by users. They are actually dedicated to 
to developers, not CLI users.

There is a global option `--traceback` flag that user can set to dump 
stack traces. You don't need to print stack traces from your code.  

### Package `xcube.core`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/core`); 
1. is covered by API documentation;
1. follows existing xcube API conventions;
1. follows PEP8 conventions;
1. is reflected in xarray extension class `xcube.core.xarray.DatasetAccessor`;
1. is reflected in CLI and WebAPI if desired;
1. is reflected in `CHANGES.md`.

#### Hints

Create new module in `xcube.core` and add your functions.
For any functions added make sure naming is in line with other API.
Add clear doc-string to the new API. Use Sphinx RST format.

Decide if your API methods requires [xcube datasets](./cubespec.md) as 
inputs, if so, name the primary dataset argument `cube` and add a 
keyword parameter `cube_asserted: bool = False`. 
Otherwise name the primary dataset argument `dataset`.

Reflect the fact, that a certain API method or function operates only 
on datasets that conform with the xcube dataset specifications by
using `cube` in its name rather than `dataset`. For example
`compute_dataset` can operate on any xarray datasets, while 
`get_cube_values_for_points` expects a xcube dataset as input or 
`read_cube` ensures it will return valid xcube datasets only. 

In the implementation, if `not cube_asserted`, 
we must assert and verify the `cube` is a cube. 
Pass `True` to `cube_asserted` argument of other API called later on: 
    
    from xcube.core.verify import assert_cube

    def frombosify_cube(cube: xr.Dataset, ..., cube_asserted: bool = False):  
        if not cube_asserted:
            assert_cube(cube)
        ...
        result = bibosify_cube(cube, ..., cube_asserted=True)
        ...

If `import xcube.core.xarray` is imported in client code, any `xarray.Dataset` 
object will have an extra property `xcube` whose interface is defined 
by the class `xcube.core.xarray.DatasetAccessor`. This class is an 
[xarray extension](http://xarray.pydata.org/en/stable/internals.html#extending-xarray) 
that is used to reflect `xcube.core` functions and make it directly 
applicable to the `xarray.Dataset` object.

Therefore any xcube API shall be reflected in this extension class.


### Package `xcube.webapi`

#### Checklist

Make sure your change

1. is covered by unit-tests (package `test/webapi`); 
1. is covered by Web API specification and documentation (currently in
   `webapi/res/openapi.yml`);
1. follows existing xcube Web API conventions;
1. follows PEP8 conventions;
1. is reflected in CLI and API, if desired;
1. is reflected in `CHANGES.md`.

### Hints

* The Web API is defined in `webapi.app` which defines mapping from 
  resource URLs to handlers
* All handlers are implemented in `webapi.handlers`. Handler code just 
  delegates to dedicated controllers.
* All controllers are implemented in `webapi.controllers.*`. They might
  further delegate into `core.*`

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
1. Create development branch named "<developer>-<issue>-<title>" 
   or "<developer>-<issue>-<title>-fix" (see below).
1. Develop, having in mind the checklists and implementation hints
   above.
   1. In your first commit, refer the issue so it will appear as link 
      in the issue history
   1. Develop, test, and push to the remote branch as desired. 
   1. In your last commit, utilize checklists above. 
      (You can include the line "closes #<issue>" in your commit message
      to auto-close the issue once the PR is merged.)
1. Create PR if build servers succeed on your branch. If not, fix issue
   first.  
   For the PR assign the team for review, agree who is to merge. 
   Also reviewers should have checklist in mind.
1. Merge PR after all reviewers are accepted your change. Otherwise go
   back. 
1. Remove issue label "in progress".
1. Delete the development branch.
1. If the PR is only partly solving an issue:
   1. Make sure the issue contains a to-do list (checkboxes) to complete
      the issue.
   1. Do not include the line "closes #<issue>" in your last commit
      message.
   1. Add "relates to issue#" in PR.
   1. Make sure to check the corresponding to-do items (checkboxes) 
      *after* the PR is merged.
   1. Remove issue label "in progress".
   1. Leave issue open.

## Branches and Releases

### Target Branches

* The `master` branch contains latest developments, including new 
  features and fixes. It is used to generate `<major>.<minor>.0` 
  releases. That is, either `<major>` or `<minor>` is increased.
* The `<major>.<minor>.x` branch is the maintenance branch for a 
  former release tagged `v<major>.<minor>.0`. It is used to generate
  maintenance `<major>.<minor>.<fix>` releases. That is, only `<fix>` 
  is increased. Most changes to `<major>.<minor>.x` branch must 
  obviously be merged into `master` branch too.
  
The software version string on all active branches is always 
`<major>.<minor>.<micro>.dev<n>`. Only for a release, we remove the 
`.dev<n>` suffix.

### Development Branches

Development branches that target the `<major>.<minor>.x` branch 
should indicate that by using the suffix `-fix`, 
e.g. `coolguy-7633-div_by_zero_in_mean-fix`. After a pull request,
the development branch will first be merged into the 
`<major>.<minor>.x` branch then into `master`.


 
