FROM mambaorg/micromamba:1.3.1

# Install xcube dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Copy files for xcube source install
COPY --chown=$MAMBA_USER:$MAMBA_USER ./xcube /tmp/xcube
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py /tmp/setup.py

# Switch into /tmp to install xcube
WORKDIR /tmp

# Required to activate env
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install xcube from source
RUN python setup.py install

# TODO: install our xcube plugins here

# micromamba entrypoint. Allows us to run container as executable
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Default command (shell form)
CMD xcube --help
