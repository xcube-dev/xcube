# Installation

## Installation from the conda-forge package

xcube releases are distributed as conda packages through the
[conda-forge](https://conda-forge.org/) channel. To install a released version
of xcube, you need a
[conda-compatible package manager](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html),
such as [Conda](https://docs.conda.io/),
[Mamba](https://mamba.readthedocs.io/),
[Micromamba](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html),
or [Pixi](https://pixi.sh/).

The following example uses Mamba to install the latest xcube release into a
new environment called `xcube`:

```bash
mamba create --name xcube --channel conda-forge xcube
```

You can give the environment a different name by providing a different argument
to the `--name` option.

To install xcube into an existing, currently activated conda-compatible
environment, use:

```bash
mamba install --channel conda-forge xcube
```

## Installation from the source code repository

Use this method if you intend to contribute to xcube or need changes that are
not yet available in an official release. A source installation requires
[Pixi](https://pixi.sh/).

First, clone the repository and install its default Pixi environment:

```bash
git clone https://github.com/xcube-dev/xcube.git
cd xcube
pixi install
```

From this point on, all instructions assume that your current directory is the
root of the xcube repository. The Pixi project configuration in
`pyproject.toml` defines the environment and installs xcube in editable mode,
so changes to the source code take effect without reinstalling the package.

You can either run commands in the environment using `pixi run`, or activate
the environment in your current shell:

```bash
pixi shell
```

To update the checkout and synchronize the environment with changes in
`pyproject.toml` and `pixi.lock`:

```bash
git pull
pixi install
```

The Pixi configuration is the source of truth for xcube's environments. If a
tool requires the legacy conda environment files, you can generate them from
the repository root:

```bash
pixi project export conda-environment environment.yml
pixi project export conda-environment --environment docs rtd-environment.yml
```

The generated files are for interoperability with conda-compatible tools and
are not tracked in the repository.

The default environment includes the development and test dependencies. Run
the unit test suite with:
    
```bash
pixi run pytest
```

To analyse test coverage:

```bash
pixi run pytest --cov=xcube
```

To produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html):

```bash
pixi run pytest --cov-report html --cov=xcube
```

## Docker

To start a demo using docker use the following commands

```bash
docker build -t [your name] .
docker run [your name]
docker run -d -p [host port]:8080 [your name]
```
    
Example 1:

```bash
docker build -t xcube:0.10.0 .
docker run xcube:0.10.0
```

This will create the docker container and list the functionality of the 
`xcube` cli.

Example 2:

```bash
docker build -t xcube:0.10.0 .
docker run -d -p 8001:8080 xcube:0.10.0 "xcube serve -v --address 0.0.0.0 --port 8080 -c /home/xcube/examples/serve/demo/config.yml"
docker ps
```

This will start a service in the background which can be accessed 
through port 8001, as the startup of a service is configured as default
behaviour.

## Installing soft dependencies

In addition to xcube’s ‘hard’ dependencies, which the standard installation
methods install automatically when required, there are ‘soft’ dependencies.
These are packages which are not required to install or use xcube, but enable
additional functionality if they are present. Users who need this functionality
should install the requisite soft dependencies explicitly. xcube’s current soft
dependencies are listed below.

- `adlfs`: required by the abfs data store, which is used for access to
  Azure Blob storage. Trying to create an abfs data store without `adlfs`
  installed will raise an exception advising that you install it.
