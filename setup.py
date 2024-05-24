#!/usr/bin/env python3

# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os

from setuptools import setup, find_packages


requirements = [
    "botocore>=1.34.51",
    "cftime>=1.6.3",
    "click>=8.0",
    "cmocean>=2.0",
    "dask>=2021.6",
    "dask-image>=0.6",
    "deprecated>=1.2",
    "distributed>=2021.6",
    "fiona>=1.8",
    "fsspec>=2021.6",
    "gdal>=3.0",
    "geopandas>=0.8",
    "jdcal>=1.4",
    "jsonschema>=3.2",
    "mashumaro",
    "matplotlib>=3.8.3",
    "netcdf4>=1.5",
    "numba>=0.52",
    "numcodecs>=0.12.1",
    "numpy>=1.16",
    "pandas>=1.3",
    "pillow>=6.0",
    "pyjwt>=1.7",
    "pyproj>=3.0",
    "pyyaml>=5.4",
    "rasterio>=1.2",
    "requests>=2.25",
    "rfc3339-validator>=0.1",
    "rioxarray>=0.11",
    "s3fs>=2021.6",
    "setuptools>=41.0",
    "shapely>=1.6",
    "tornado>=6.0",
    "urllib3>=1.26",
    "xarray>=2022.6",
    "zarr>=2.11"
]

packages = find_packages(exclude=["test", "test.*"])

# Same effect as "from cate import version", but avoids importing cate:
version = None
with open('xcube/version.py') as f:
    exec(f.read())

# noinspection PyTypeChecker
setup(
    name=os.getenv("XCUBE_PYPI_NAME", "xcube"),
    version=version,
    description=('xcube is a Python package for generating and exploiting '
                 'data cubes powered by xarray, dask, and zarr.'),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license='MIT',
    author='xcube Development Team',
    packages=packages,
    package_data={
        'xcube.webapi.meta': [
            'data/openapi.html'
        ],
        'xcube.webapi.viewer': [
            'data/*', 'data/**/*'
        ]
    },
    entry_points={
        'console_scripts': [
            # xcube's CLI
            'xcube = xcube.cli.main:main',
        ],
        'xcube_plugins': [
            # xcube's default extensions
            'xcube = xcube.plugin:init_plugin',
        ],
    },
    install_requires=requirements,
    # these links will be shown in the left panel on PyPI
    project_urls={
        "Documentation": "https://xcube.readthedocs.io/en/latest/",
        "Source": "https://github.com/xcube-dev/xcube",
        "Download": "https://pypi.org/project/xcube-core/#files",
        "Tracker": "https://github.com/xcube-dev/xcube/issues",
        "Release notes": "https://github.com/xcube-dev/xcube/releases",
        "Changelog": "https://github.com/xcube-dev/xcube/blob/main/CHANGES.md"
    },
    # these classifiers will be shown in the left panel on PyPI
    # they to not interfer with pip install and are meta data for the user
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ]
)
