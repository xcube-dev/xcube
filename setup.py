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
    #
    # xcube requirements are given in file ./environment.yml.
    #
    # All packages here have been commented out,
    # because otherwise setuptools will install
    # additional pip packages although conda packages
    # with same name are already available
    # in the conda environment defined by file ./environment.yml.
    #
    # 'affine',
    # 'click',
    # 'cmocean',
    # 'dask',
    # 'fiona',
    # 'gdal',
    # 'matplotlib',
    # 'netcdf4',
    # 'numba',
    # 'numpy',
    # 'pandas',
    # 'pillow',
    # 'proj4',
    # 'pyyaml',
    # 'rasterio',
    # 's3fs',
    # 'setuptools',
    # 'shapely',
    # 'tornado',
    # 'xarray',
    # 'zarr',
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
)
