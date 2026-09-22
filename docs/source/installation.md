# Installation

## Installation from the conda-forge package

xcube releases are distributed as conda packages through the
[conda-forge](https://conda-forge.org/) channel. To install a released version
of xcube, you need a
[conda-compatible package manager](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html),
such as [Conda](https://docs.conda.io/),
[Mamba](https://mamba.readthedocs.io/),
[Micromamba](https://mamba.readthedocs.io/en/stable/user_guide/micromamba.html),
or [Pixi](https://pixi.sh/).

The following example uses Mamba to install the latest xcube release into a
new environment called `xcube`:

```bash
mamba create --name xcube --channel conda-forge xcube
```

You can give the environment a different name by providing a different argument
to the `--name` option.

To install xcube into an existing, currently activated conda-compatible
environment, use:

```bash
mamba install --channel conda-forge xcube
```

## Installation from the source code repository

Use this method if you intend to contribute to xcube or need changes that are
not yet available in an official release. A source installation requires
[Pixi](https://pixi.sh/).

First, clone the repository and install its default Pixi environment:

```bash
git clone https://github.com/xcube-dev/xcube.git
cd xcube
pixi install
```

From this point on, all instructions assume that your current directory is the
root of the xcube repository. The Pixi project configuration in
`pyproject.toml` defines the environment and installs xcube in editable mode,
so changes to the source code take effect without reinstalling the package.

You can either run commands in the environment using `pixi run`, or activate
the environment in your current shell:

```bash
pixi shell
```

To update the checkout and synchronize the environment with changes in
`pyproject.toml` and `pixi.lock`:

```bash
git pull
pixi install
```

The Pixi configuration is the source of truth for xcube's environments. If a
tool requires the legacy conda environment files, you can generate them from
the repository root:

```bash
pixi project export conda-environment --from-lock-file environment.yml
pixi project export conda-environment --from-lock-file --environment docs rtd-environment.yml
```

The generated files are for interoperability with conda-compatible tools and
are not tracked in the repository.

The default environment includes the development and test dependencies. Run
the unit test suite with:
    
```bash
pixi run tests
```

To run the tests with coverage and produce an HTML
[coverage report](https://pytest-cov.readthedocs.io/en/latest/reporting.html)
in `.cov-report/`:

```bash
pixi run coverage
```

## Docker images

The xcube repository contains a Dockerfile which can be used to build a
Docker image of xcube. A pre-built Docker image is also published for each
release.

### Pulling pre-built xcube Docker images

xcube Docker images are published at `quay.io/xcube` and tagged with the
version number, prefixed with `v`. So, for instance, you can pull an image
for version 1.14.1 of xcube with this command:

```bash
docker pull quay.io/bcdev/xcube:v1.14.1
```

### Building your own xcube Docker images

You can build an xcube Docker image from the repository. This can be useful
if you require an image with customizations not available in the published
release images. To build and tag a new xcube image, clone the git repository
and execute the following command in the root directory of the repository:

```bash
docker build -t [identifier] .
```

The format of the identifier is `[registry/][repository][:tag]`.
The registry, repository, and tag can be freely chosen, and the registry and
tag can be omitted. For example:

```bash
docker build -t xcube-custom:1.13.3 .
```

### Running xcube from a Docker image

To run the default command in a Docker image, use this command:

```bash
docker run [identifier]
```

For example, to use version 1.14.1 of the published xcube Docker image:

```bash
docker run quay.io/bcdev/xcube:v1.14.1
```

The default command in the published images is `xcube --help`, so running
this command will simply output the help message for the `xcube` CLI command.

For a more interesting demonstration, you can build a customized image and
use it to run a local xcube server with some example datasets. In the
root directory of the xcube git repository, run the following command.

```bash
echo 'COPY --chown=$MAMBA_USER:$MAMBA_USER examples' \
     '/home/$MAMBA_USER/examples' >>Dockerfile
```

The command above adds a line to the Dockerfile which copies some example
configurations and data into the container image during the build process.
Now build your customized image:

```bash
docker build -t xcube-custom:1 .
```

Now you have a local xcube Docker image which you can run as a server:

```bash
docker run -d -p 8080:8080 xcube-custom:1 xcube serve -v --address 0.0.0.0 \
       -c /home/xcube/examples/serve/demo/config.yml
```

This will start an xcube server in the background. You can see details of the
running process like this:

```bash
docker ps
```

Now you can use a web browser to interact with the xcube server:

-   <http://localhost:8080/viewer/> to explore the example datasets using
    xcube's web viewer
-   <http://localhost:8080/openapi.html> to explore the xcube server's
    REST APIs from its OpenAPI page

## Installing soft dependencies

In addition to xcube’s ‘hard’ dependencies, which the standard installation
methods install automatically when required, there are ‘soft’ dependencies.
These are packages which are not required to install or use xcube, but enable
additional functionality if they are present. Users who need this functionality
should install the requisite soft dependencies explicitly. xcube’s current soft
dependencies are listed below.

- `adlfs`: required by the abfs data store, which is used for access to
  Azure Blob storage. Trying to create an abfs data store without `adlfs`
  installed will raise an exception advising that you install it.
