# Image from https://hub.docker.com (syntax: repo/image:version)
FROM continuumio/miniconda3:latest

ARG INSTALL_PLUGINS=1

# Metadata
LABEL maintainer="helge.dzierzon@brockmann-consult.de"
LABEL name=xcube
LABEL version=0.9.0.dev0
LABEL conda_env=xcube

# Ensure usage of bash (ensures conda calls succeed)
SHELL ["/bin/bash", "-c"]

# Update system and install dependencies
RUN apt-get -y update && apt-get -y upgrade
# Allow editing files in container
RUN apt-get -y install vim

# Install mamba as a much faster conda replacement. We specify an
# explicit version number because (1) it makes installation of mamba
# much faster and (2) mamba is still in beta, so it's best to stick
# to a known-good version.
RUN conda install -c conda-forge mamba

# Setup conda environment
# Copy yml config into image
COPY environment.yml /tmp/environment.yml

# Use mamba to create an environment based on the specifications in
# environment.yml. 
RUN mamba env create --file /tmp/environment.yml

# Set work directory for xcube installation
RUN mkdir /xcube
WORKDIR /xcube

# Copy sources into xcube
COPY . /xcube

# Setup xcube package.
RUN source activate xcube && python setup.py install

WORKDIR /tmp
ADD scripts/install_xcube-datastore.sh ./

ENV XCUBE_SH_VERSION=0.8.0
RUN bash install_xcube-datastore.sh xcube-sh ${XCUBE_SH_VERSION} release
ENV XCUBE_CCI_VERSION=0.8.1.dev3
RUN bash install_xcube-datastore.sh xcube-cci ${XCUBE_CCI_VERSION} release
ENV XCUBE_CDS_VERSION=0.8.1
RUN bash install_xcube-datastore.sh xcube-cds ${XCUBE_CDS_VERSION} release

# Export web server port
EXPOSE 8080

# Start bash, so we can invoke xcube CLI.
ENTRYPOINT ["/bin/bash", "-c"]
# By default, activate xcube environment and print usage help.
CMD ["source activate xcube && xcube serve -v --address 0.0.0.0 --port 8080 -c ./examples/serve/demo/config.yml"]
