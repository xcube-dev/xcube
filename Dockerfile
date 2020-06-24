# Image from https://hub.docker.com (syntax: repo/image:version)
FROM continuumio/miniconda3:latest

# Person responsible
MAINTAINER helge.dzierzon@brockmann-consult.de

LABEL name=xcube
LABEL version=0.3.0.dev1
LABEL conda_env=xcube

# Ensure usage of bash (simplifies source activate calls)
SHELL ["/bin/bash", "-c"]

# Update system and install dependencies
RUN apt-get -y update && apt-get -y upgrade

# && apt-get -y install  git build-essential libyaml-cpp-dev

# Install mamba as a much faster conda replacement. We specify an
# explicit version number because (1) it makes installation of mamba
# much faster and (2) mamba is still in beta, so it's best to stick
# to a known-good version.
RUN conda install mamba=0.1.2 -c conda-forge

# Setup conda environment
# Copy yml config into image
ADD environment.yml /tmp/environment.yml

# Use mamba to create an environment based on the specifications in
# environment.yml. At present, evironments created by mamba can't be
# referenced by name from conda (presumably a bug), so we use --preix
# to specify an explicit path instead.
RUN mamba env create --file /tmp/environment.yml

# Set work directory for xcube_server installation
RUN mkdir /xcube
WORKDIR /xcube

# Copy local github repo into image (will be replaced by either git clone or as a conda dep)
ADD . /xcube

# Setup xcube_server package, specifying the environment by path rather
# than by name (see above).
RUN source activate xcube && python setup.py develop

# Test xcube package
ENV NUMBA_DISABLE_JIT 1
RUN source activate xcube && pytest

# Export web server port 8000
EXPOSE 8000

# Start server
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source activate xcube && xcube --help"]
