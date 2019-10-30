# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import xcube.util.ext


def init_plugin(ext_registry: xcube.util.ext.ExtensionRegistry):
    """
    xcube dataset I/O standard extensions
    """
    ext_registry.add_ext_lazy(_load_dsio_zarr,
                              'xcube.core.dsio', 'zarr',
                              description='Zarr file format (http://zarr.readthedocs.io)')
    ext_registry.add_ext_lazy(_load_dsio_netcdf4,
                              'xcube.core.dsio', 'netcdf4',
                              description='NetCDF-4 file format')
    ext_registry.add_ext_lazy(_load_dsio_csv,
                              'xcube.core.dsio', 'csv',
                              description='CSV file format')
    ext_registry.add_ext_lazy(_load_dsio_mem,
                              'xcube.core.dsio', 'mem',
                              description='In-memory dataset I/O')


def _load_dsio_zarr():
    from xcube.util.dsio import ZarrDatasetIO
    return ZarrDatasetIO()


def _load_dsio_netcdf4():
    from xcube.util.dsio import Netcdf4DatasetIO
    return Netcdf4DatasetIO()


def _load_dsio_csv():
    from xcube.util.dsio import CsvDatasetIO
    return CsvDatasetIO()


def _load_dsio_mem():
    from xcube.util.dsio import MemDatasetIO
    return MemDatasetIO()
