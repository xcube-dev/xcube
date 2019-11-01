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


from xcube.util import extension


def init_plugin(ext_registry: extension.ExtensionRegistry):
    """
    xcube dataset I/O standard extensions
    """
    ext_registry.add_extension(loader=extension.import_component('xcube.util.dsio:ZarrDatasetIO', call=True),
                               point='xcube.core.dsio', name='zarr',
                               description='Zarr file format (http://zarr.readthedocs.io)',
                               ext='zarr', modes={'r', 'w', 'a'})
    ext_registry.add_extension(loader=extension.import_component('xcube.util.dsio:Netcdf4DatasetIO', call=True),
                               point='xcube.core.dsio', name='netcdf4',
                               description='NetCDF-4 file format',
                               ext='nc', modes={'r', 'w', 'a'})
    ext_registry.add_extension(loader=extension.import_component('xcube.util.dsio:CsvDatasetIO', call=True),
                               point='xcube.core.dsio', name='csv',
                               description='CSV file format',
                               ext='csv', modes={'r', 'w'})
    ext_registry.add_extension(loader=extension.import_component('xcube.util.dsio:MemDatasetIO', call=True),
                               point='xcube.core.dsio', name='mem',
                               description='In-memory dataset I/O',
                               ext='mem', modes={'r', 'w', 'a'})
