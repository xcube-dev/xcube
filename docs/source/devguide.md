# xcube Developer Guide

Version 0.2, draft

*IMPORTANT NOTE: Any changes to this doc must be reviewed by dev-team
through pull requests.*


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

We follow [PEP-8](https://www.python.org/dev/peps/pep-0008/), including
its recommendation of [PEP-484](https://www.python.org/dev/peps/pep-0484/)
syntax for type hints.

### Updating code style in the existing codebase

A significant portion of the existing codebase does not adhere to our current
code style guidelines. It is of course a goal to bring these parts into
conformance with the style guide, but major style changes should not be
bundled into pull requests focused on other improvements or bug fixes, because
they obscure the significant code changes and make reviews difficult.
Large-scale style and formatting updates should instead be made via dedicated
pull requests.

### Line length

As recommended in PEP-8, all lines should be limited to a maximum of 79
characters, including docstrings and comments.

### Quotation marks for string literals

In general, single quotation marks should always be used for string literals.
Double quotation marks should only be used if there is a compelling reason to
do so in a particular case.

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
1. follows our [code style](#coding-style) conventions;
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
1. follows our [code style](#coding-style) conventions;
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
1. follows our [code style](#coding-style) conventions;
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

1. Make sure there is an issue ticket for your code change work item.
1. Select issue; priorities are as follows:
   1. "urgent" and ("important" and "bug")
   1. "urgent" and ("important" or "bug")
   1. "urgent"
   1. "important" and "bug"
   1. "important" or "bug"
   1. others
1. Make sure issue is assigned to you; if unclear, agree with team first.
1. Add issue label "in progress".
1. Clone the GitHub repository to your local machine, if you haven't
   already done so.
1. Check out the `master` branch and make sure it's up to date (`git pull`).
1. Create a development branch named `"<developer>-<issue>-<title>"` 
   (see [below](#development-branches)).
1. Develop, having in mind the checklists and implementation hints
   above.
   1. In your first commit, refer the issue so it will appear as link 
      in the issue history.
   1. It is useful to push your first commit to the GitHub repository
      at once and create a draft pull request for your branch, to make
      it easier to find online. Also, it will run continuous integration and
      check the compatibility of your branch with the main branch.
   1. Develop, test, and push to the remote branch as desired. 
   1. In your last commit, utilize checklists above. 
      (You can include the line "closes #`<issue>`" in your commit message
      to auto-close the issue once the PR is merged.)
1. When your branch is ready to merge (as determined by the checklist), either
   create a pull request, or, if you had created a draft in the previous step,
   remove its draft status and invite one or more reviewers. Exactly who
   should review depends on how potentially dangerous your changes are, who's
   available, who knows that part of the codebase well, etc. – if unsure,
   discuss with the team. You can also mention other team members with the
   `@username` syntax in a PR comment to make sure that they're aware of the
   PR even if they don't need to review it.
1. Wait for reviewers to review your PR and respond to comments and
   suggestions. Reviewers can also commit to your branch with additional
   changes.
1. Once all reviewers have given an "accept" review, the PR should be
   merged. The last reviewer to give an "accept" should perform the merge;
   if they forget, the PR author should merge.
1. Remove issue label "in progress".
1. Delete the feature branch.
1. If the PR completely solves the associated issue: close the issue, if it
   wasn't already automatically closed by a keyword in the pull request
   description, and make sure that there is a link from the issue to the
   PR (e.g. by adding a "fixed by PR #123" comment on the issue).
1. If the PR is only partly solving an issue:
   1. Make sure the issue contains a to-do list (checkboxes) to complete
      the issue.
   1. Do not include the line "closes #`<issue>`" in your last commit
      message.
   1. Add "relates to issue#" in PR.
   1. Make sure to check the corresponding to-do items (checkboxes) 
      *after* the PR is merged.
   1. Remove issue label "in progress".
   1. Leave issue open.

## Branches and Releases

### Target Branch

The `master` branch contains latest developments,
including new features and fixes.
Its software version string is always `<major>.<minor>.<micro>.dev<n>`.
The branch is used to generate major, minor, or maintenance releases.
That is, either `<major>`, `<minor>`, or `<fix>` is increased.
Before a release, the last thing we do is to remove the `.dev<n>` suffix, 
after a release, the first thing we do is to increase the `micro` version
and add the `.dev<n>` suffix.

### Development Branches

Development branches should be named `<developer>-<issue>-<title>` where
* `<developer>` is the github name of the code author
* `<issue>` is the number of the issue in the github issue tracker that is
  targeted by the work on this branch
* `<title>` is either the name of the issue or an abbreviated version of it

## Release Process

### Release on GitHub

This describes the release process for `xcube`. For a plugin release, 
you need to adjust the paths accordingly. 

* Check issues in progress, close any open issues that have been fixed.
* Make sure that all unit tests pass and that test coverage is 100% 
  (or as near to 100% as practicable).
* In `xcube/version.py` remove the `.dev` suffix from version name.
* Adjust version in `Dockerfile`  accordingly.
* Make sure `CHANGES.md` is complete. Remove the suffix ` (in development)` 
  from the last version headline.
* Push changes to either master or a new maintenance branch (see above).
* Await results from Travis CI and ReadTheDocs builds. If broken, fix.
* Go to [xcube/releases](https://github.com/dcs4cop/xcube/releases) 
  and press button "Draft a new Release".
  - Tag version is: `v${version}` (with a "v" prefix)
  - Release title is: `${version}` (without a "v" prefix) 
  - Paste latest changes from `CHANGES.md` into field "Describe this release"
  - Press "Publish release" button
* After the release on GitHub, rebase sources, if the branch was `master`, 
  create a new maintenance branch (see above)
* In `xcube/version.py` increase version number and append a `.dev0` suffix 
  to the version name so that it is still PEP-440 compatible.
* Adjust version in `Dockerfile` accordingly.
* In `CHANGES.md` add a new version headline and attach ` (in development)`
  to it.
* Push changes to either master or a new maintenance branch (see above).
* Activate new doc version on ReadTheDocs. 

Go through the same procedure for all xcube plugin packages 
dependent on this version of xcube.

### Release on Conda-Forge

These instructions are based on the documentation at 
[conda-forge](https://conda-forge.org/docs/maintainer/updating_pkgs.html).

Conda-forge packages are produced from a github feedstock repository belonging 
to the conda-forge organization. A repository's feedstock is usually located at 
`https://github.com/conda-forge/<repo-name>-feedstock`, e.g., 
`https://github.com/conda-forge/xcube-feedstock`.

Usually, a conda-forge bot will create a feedstock pull request automatically for each new GitHub release, and the maintainers only need to merge it (potentially after some manual improvements). The manual procedure below is only needed if there isn't time to wait for the bot, or if an additional conda-forge build has to be done without a new GitHub release.
The package is updated by 
* forking the repository
* creating a new branch for the changes
* creating a pull request to merge this branch into conda-forge's feedstock repository 
  (this is done automatically if the build number is 0). 

In detail, the steps are:

1. Fork `https://github.com/conda-forge/<repo-name>-feedstock` in your personal GitHub
   account.

1. Clone the repository locally and create a new branch. The name of the branch 
   is not strictly prescribed, but it's sensible to choose an informative name like 
   `update_0_5_3`.

1. In case the build number is 0, a bot will render the feedstock during the pull request.
   Otherwise, conduct the following steps: 
   Rerender the feedstock using conda-smithy. This updates common conda-forge 
   feedstock files. It's probably easiest to install conda-smithy in a 
   fresh environment for this:

   `conda install -c conda-forge conda-smithy`
    
   `conda smithy rerender -c auto`

1. Update `recipe/meta.yaml` for the new version. 
   Mainly this will involve the following steps:

   1. Update the value of the version variable (or, if the version number 
   has not changed, increment the build number).
   
   1. If the version number has changed, ensure that the build number is set to 0.

   1. Update the sha256 hash of the source archive prepared by GitHub.

   1. If the dependencies have changed, update the list of dependencies 
   in the `-run` subsection to match those in the environment.yml file.

1. Commit the changes and push them to GitHub. 
   A pull request at the feedstock repository on conda-forge will be automatically 
   created by a bot if the build number is 0.
   If it is higher, you will have to create the pull request yourself.

1. Once conda-forge's automated checks have passed, merge the pull request.

1. Delete your personal fork of the `xcube-feedstock`-repository.

Once the pull request has been merged, the updated package should usually become 
available from conda-forge within a couple of hours.

TODO: Describe deployment of xcube Docker image after release

If any changes apply to `xcube serve` and the xcube Web API:

Make sure changes are reflected in `xcube/webapi/res/openapi.yml`. 
If there are changes, sync `xcube/webapi/res/openapi.yml` with 
xcube Web API docs on SwaggerHub.

Check if changes affect the xcube-viewer code. If so
make sure changes are reflected in xcube-viewer code and 
test viewer with latest xcube Web API. Then release a new xcube viewer.

### xcube Viewer 

* Cd into viewer project directory (`.../xcube-viewer/.`).
* Remove the `-dev` suffix from `version` property in `package.json`.
* Remove the `-dev` suffix from `version` constant in `src/config.ts`.
* Make sure `CHANGES.md` is complete. Remove the suffix ` (in development)` 
  from the last version headline.
* Build the app and test the build using a local http-server, e.g.:

    $ npm install -g http-server
    $ cd build 
    $ http-server -p 3000 -c-1

* Push changes to either master or a new maintenance branch (see above).
* Goto [xcube-viewer/releases](https://github.com/dcs4cop/xcube-viewer/releases) 
  and press button "Draft a new Release".
  - Tag version is: `v${version}` (with a "v" prefix).
  - Release title is: `${version}`. 
  - Paste latest changes from `CHANGES.md` into field "Describe this release".
  - Press "Publish release" button.
* After the release on GitHub, if the branch was `master`, 
  create a new maintenance branch (see above).
* Increase `version` property and `version` constant in `package.json` and `src/config.ts` 
  and append `-dev.0` suffix to version name so it is SemVer compatible.
* In `CHANGES.md` add a new version headline and attach ` (in development)` to it.
* Push changes to either master or a new maintenance branch (see above).
* Deploy builds of `master` branches to related web content providers.
