FROM quay.io/bcdev/xcube-python-base:0.8.1

ARG INSTALL_PLUGINS=1

# Metadata
LABEL maintainer="helge.dzierzon@brockmann-consult.de"
LABEL name=xcube
LABEL version=0.9.0.dev0
LABEL conda_env=xcube

# Ensure usage of bash (ensures conda calls succeed)
SHELL ["/bin/bash", "-c"]

USER root
# Update system for security checks
RUN apt-get -y update && apt-get -y upgrade

USER xcube

# Setup conda environment
# Copy yml config into image
COPY environment.yml /tmp/environment.yml

# Use mamba to create an environment based on the specifications in
# environment.yml. 
RUN mamba env create --file /tmp/environment.yml

# Set work directory for xcube installation
WORKDIR /home/xcube

# Copy sources into xcube
COPY . ./

# Setup xcube package.
RUN source activate xcube && python setup.py install

WORKDIR /tmp
ADD scripts/install_xcube-datastore.sh ./

ENV XCUBE_SH_VERSION=0.8.0
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube-datastore.sh xcube-sh ${XCUBE_SH_VERSION} release; fi;
ENV XCUBE_CCI_VERSION=0.8.1.dev3
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube-datastore.sh xcube-cci ${XCUBE_CCI_VERSION} release; fi;
ENV XCUBE_CDS_VERSION=0.8.1
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube-datastore.sh xcube-cds ${XCUBE_CDS_VERSION} release; fi;

# Export web server port
EXPOSE 8080

# Start bash, so we can invoke xcube CLI.
CMD ["/bin/bash"]
