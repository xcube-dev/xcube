ARG MINICONDA_VERSION=latest

FROM continuumio/miniconda3:${MINICONDA_VERSION}

ARG INSTALL_PLUGINS=1
ARG XCUBE_USER_NAME=xcube
ENV XCUBE_SH_VERSION=latest
ENV XCUBE_CCI_VERSION=latest
ENV XCUBE_CDS_VERSION=latest

# Metadata
LABEL maintainer="xcube-team@brockmann-consult.de"
LABEL name=xcube
LABEL conda_env=xcube

# Ensure usage of bash (ensures conda calls succeed)
SHELL ["/bin/bash", "-c"]

USER root
# Update system for security checks
RUN apt-get -y update && apt-get -y upgrade vim jq curl

SHELL ["/bin/bash", "-c"]
RUN groupadd -g 1000 ${XCUBE_USER_NAME}
RUN useradd -u 1000 -g 1000 -ms /bin/bash ${XCUBE_USER_NAME}
RUN mkdir /workspace && chown ${XCUBE_USER_NAME}:${XCUBE_USER_NAME} /workspace
RUN chown -R ${XCUBE_USER_NAME}:${XCUBE_USER_NAME} /opt/conda

USER ${XCUBE_USER_NAME}

RUN source activate base && conda update -n base conda && conda init
RUN source activate base && conda install -n base -c conda-forge mamba==0.24.0 pip=21.3.1

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
ADD scripts/install_xcube.sh ./

RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube.sh xcube-sh ${XCUBE_SH_VERSION} release; fi;
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube.sh xcube-cci ${XCUBE_CCI_VERSION} release; fi;
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install_xcube.sh xcube-cds ${XCUBE_CDS_VERSION} release; fi;

# Export web server port
EXPOSE 8080

# Run bash in xcube environment, so we can invoke xcube CLI.
ENTRYPOINT ["conda", "run", "-v", "-n", "xcube", "/bin/bash", "-c"]

# By default show xcube help 
CMD ["xcube --help"]
