# Installation

## Prerequisites

xcube releases are packaged as conda packages in the
[conda-forge](https://conda-forge.org/) channel. It is recommended to install
xcube into a conda environment using the
[mamba package manager](https://github.com/mamba-org/mamba), which will
also automatically install and manage xcube’s dependencies.
You can find [installation instructions for mamba itself
here](https://mamba.readthedocs.io/en/latest/installation.html), if you
don’t already have it installed.

In addition to mamba, there are alternative package managers available for
conda environments:

1. The original `conda` tool. When considering this tool, please note that
   package installation and management with conda may be significantly slower 
   than with mamba.
2. The `micromamba` tool, a minimalistic, self-contained version of mamba.

## Overview of installation methods

There are two main ways to install the xcube package:

1.  **Install an official release** from a conda-forge package, using the mamba
    package manager. This method is recommended for most users.
2.  Use mamba to install only xcube’s dependencies, but not xcube itself.
    Then **clone the xcube git repository** and install directly from your
    local repository. You should use this method if you intend to participate
    in the development of xcube, or if you need to use features that are
    so new that they are not yet available in an officially release conda-forge
    package.

These methods are described in more detail in the following sections.

## Installation from the conda-forge package

To install the latest release of xcube into a new conda environment called
`xcube`, run the following command.

```bash
mamba create --name xcube --channel conda-forge xcube
```

You can give the environment a different name by providing a different argument
to the `--name` option.

To install xcube into an existing, currently activated conda environment,
use the following command.

```bash
mamba install --channel conda-forge xcube
```

## Installation from the source code repository

First, clone the repository and create a conda environment from it:

```bash
git clone https://github.com/dcs4cop/xcube.git
cd xcube
mamba env create
```

From this point on, all instructions assume that your current directory is the
root of the xcube repository.

The `mamba env create` command above creates an environment according to
the specifications in the `environment.yml` file in the repository, which
by default takes the name `xcube`. Then, to activate the environment and
install xcube from the repository:

```bash    
mamba activate xcube
pip install --no-deps --editable .
```

The second command installs xcube in ‘editable mode’, meaning that it will
be run directly from the repository, and changes to the code in the repository
will take immediate effect without reinstallation. (As an alternative to
pip, the command `python setup.py develop` can be used, but this is
[no longer recommended](https://docs.python.org/3/install/#introduction).
Among other things, `pip` has the advantage of allowing easy deinstallation of
installed packages.)

To update the install to the latest repository version and update the
environment to reflect to any changes in `environment.yml`:

```bash
mamba activate xcube
git pull --force
mamba env update -n xcube --file environment.yml --prune
```

To install `pytest` and run the unit test suite:
    
```bash
mamba install pytest
pytest
```

To analyse test coverage (after installing pytest as above):

```bash
pytest --cov=xcube
```

To produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html):

```bash
pytest --cov-report html --cov=xcube
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
