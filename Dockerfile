# Image from https://hub.docker.com (syntax: repo/image:version)
FROM continuumio/miniconda3:latest

# Person responsible
MAINTAINER helge.dzierzon@brockmann-consult.de

LABEL name=xcube
LABEL version=0.2.0.dev2
LABEL conda_env=xcube

# Ensure usage of bash (simplifies source activate calls)
SHELL ["/bin/bash", "-c"]

# Update system and install dependencies
RUN apt-get -y update && apt-get -y upgrade

# && apt-get -y install  git build-essential libyaml-cpp-dev

# Setup conda environment
# Copy yml config into image
ADD environment.yml /tmp/environment.yml

# Update conda and install dependencies specified in environment.yml
RUN conda env create -f=/tmp/environment.yml;

# Set work directory for xcube_server installation
RUN mkdir /xcube
WORKDIR /xcube

# Copy local github repo into image (will be replaced by either git clone or as a conda dep)
ADD . /xcube

# Setup xcube_server package
RUN source activate xcube && python setup.py develop

# Test xcube package
ENV NUMBA_DISABLE_JIT 1
#RUN source activate xcube && pytest

# Export web server port 8000
EXPOSE 8000

# Start server
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source activate xcube && xcube"]
