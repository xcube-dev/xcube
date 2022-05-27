# Installation

xcube can be installed from a released conda package, or directly from a
copy of the source code repository.

The first two sections below give instructions for installation using conda,
available as part of the [miniconda
distribution](https://docs.conda.io/en/latest/miniconda.html). If installation
using conda proves to be unacceptably slow, mamba can be used instead (see
[Installation using mamba](#installation-using-mamba)).

## Installation from the conda package

Into a currently active, existing conda environment (>= Python 3.7)

    $ conda install -c conda-forge xcube

Into a new conda environment named `xcube`:
    
    $ conda create -c conda-forge -n xcube xcube

The argument to the `-n` option can be changed to create a differently
named environment.

## Installation from the source code repository

First, clone the repository and create a conda environment from it:
    
    $ git clone https://github.com/dcs4cop/xcube.git
    $ cd xcube
    $ conda env create

From this point on, all instructions assume that your current directory is the
root of the xcube repository.

The `conda env create` command above creates an environment according to
the specifications in the `environment.yml` file in the repository, which
by default takes the name `xcube`. Then, to activate the environment and
install xcube from the repository:
    
    $ conda activate xcube
    $ pip install --no-deps --editable .

The second command installs xcube in ‘editable mode’, meaning that it will
be run directly from the repository, and changes to the code in the repository
will take immediate effect without reinstallation. (As an alternative to
pip, the command `python setup.py develop` can be used, but this is
[no longer recommended](https://docs.python.org/3/install/#introduction).
Among other things, `pip` has the advantage of allowing easy deinstallation of
installed packages.)

To update the install to the latest repository version and update the
environment to reflect to any changes in `environment.yml`:
    
    $ conda activate xcube
    $ git pull --force
    $ conda env update -n xcube --file environment.yml --prune
    
To install `pytest` and run the unit test suite:
    
    $ conda install pytest
    $ pytest
    
To analyse test coverage (after installing pytest as above):

    $ pytest --cov=xcube

To produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html):

    $ pytest --cov-report html --cov=xcube

## Installation using mamba

[Mamba](https://github.com/mamba-org/mamba) is a dramatically faster drop-in
replacement for the conda tool. Mamba itself can be installed using conda.
If installation using conda proves to be unacceptably slow, it is recommended
to install mamba, as follows:

    $ conda create -n xcube python=3.8
    $ conda activate xcube
    $ conda install -c conda-forge mamba

This creates a conda environment called `xcube`, activates the environment,
and installs mamba in it. To install xcube from its conda-forge package, you
can now use:

    $ mamba install -c conda-forge xcube

Alternatively, to install xcube directly from the repository:

    $ git clone https://github.com/dcs4cop/xcube.git
    $ cd xcube
    $ mamba env create
    $ pip install --no-deps --editable .

## Docker

To start a demo using docker use the following commands

    $ docker build -t [your name] .
    $ docker run [your name]
    

$ docker run -d -p [host port]:8080 [your name]
    
Example 1:

    $  docker build -t xcube:0.10.0 .
    $  docker run xcube:0.10.0

This will create the docker container and list the functionality of the 
`xcube` cli.

Example 2:

    $  docker build -t xcube:0.10.0 .
    $  docker run -d -p 8001:8080 "xcube:0.10.0 xcube serve -v --address 0.0.0.0 --port 8080 -c /home/xcube/examples/serve/demo/config.yml"
    $  docker ps

This will have started a service in the background which can be accessed 
through port 8001, as the startup of a service is configured as default
behaviour.
