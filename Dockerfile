# For micromamba image documentation,
# goto https://hub.docker.com/r/mambaorg/micromamba
ARG MICROMAMBA_VERSION=1.3.1
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

ARG NEW_MAMBA_USER=xcube
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000

ARG INSTALL_PLUGINS=0

ENV XCUBE_SH_VERSION=latest
ENV XCUBE_CCI_VERSION=latest
ENV XCUBE_CDS_VERSION=latest
ENV XCUBE_CMEMS_VERSION=latest

LABEL maintainer="xcube-team@brockmann-consult.de"
LABEL name=xcube

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
USER root

# Update system and ensure that basic commands are available.
RUN apt-get -y update && \
    apt-get -y upgrade vim jq curl wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Magic taken from https://hub.docker.com/r/mambaorg/micromamba,
# section "Changing the user id or name"
RUN usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
             "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :

ENV MAMBA_USER=$NEW_MAMBA_USER
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

USER $MAMBA_USER

# Install xcube dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Copy files for xcube source install
COPY --chown=$MAMBA_USER:$MAMBA_USER ./xcube /tmp/xcube
COPY --chown=$MAMBA_USER:$MAMBA_USER ./pyproject.toml /tmp/pyproject.toml
COPY --chown=$MAMBA_USER:$MAMBA_USER ./README.md /tmp/README.md

# Switch into /tmp to install xcube.
WORKDIR /tmp

# Required to activate env during image build.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install xcube from source.
RUN pip install --no-deps .

# Install our known xcube plugins.
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/install-xcube-plugin.sh ./
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install-xcube-plugin.sh xcube-sh ${XCUBE_SH_VERSION} release; fi;
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install-xcube-plugin.sh xcube-cci ${XCUBE_CCI_VERSION} release; fi;
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install-xcube-plugin.sh xcube-cds ${XCUBE_CDS_VERSION} release; fi;
RUN if [[ ${INSTALL_PLUGINS} == '1' ]]; then bash install-xcube-plugin.sh xcube-cmems ${XCUBE_CMEMS_VERSION} release; fi;

RUN micromamba clean --all --force-pkgs-dirs --yes

WORKDIR /home/$MAMBA_USER

# The micromamba entrypoint.
# Allows us to run container as an executable with
# base environment activated.
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Default command (shell form)
CMD xcube --help
