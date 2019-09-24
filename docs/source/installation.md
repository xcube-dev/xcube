
# Installation

## Installation using conda

Into existing conda environment (>= Python 3.7)

    $ git install -c conda-forge xcube

Into new conda environment
    
    $ git create -c conda-forge -n xcube python3
    $ git install -c conda-forge xcube


## Installation from sources

First
    
    $ git clone https://github.com/dcs4cop/xcube.git
    $ cd xcube
    $ conda env create
    
Then
    
    $ activate xcube
    $ python setup.py develop

Update
    
    $ activate xcube
    $ git pull --force
    $ python setup.py develop
    
    
Run tests

    $ pytest
    
with coverage

    $ pytest --cov=xcube

with [coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html) in HTML

    $ pytest --cov-report html --cov=xcube


## Docker

To start a demo using docker use the following commands

    $ docker build -t [your name] .
    $ docker run -d -p [host port]:8000 [your name]
    
Example:

    $  docker build -t xcube:0.1.0dev6 .
    $  docker run -d -p 8001:8000 xcube:0.1.0dev6
    $  docker ps


