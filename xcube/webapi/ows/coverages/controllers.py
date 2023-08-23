# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from xcube.webapi.datasets.context import DatasetsContext
import xarray as xr


def get_coverage(ctx: DatasetsContext, collection_id: str):
    ml_dataset = ctx.get_ml_dataset(collection_id)
    return ml_dataset


def get_coverage_domainset(ctx: DatasetsContext, collection_id: str):
    ml_dataset = ctx.get_ml_dataset(collection_id)
    ds = ml_dataset.get_dataset(0)
    assert(isinstance(ds, xr.Dataset))
    grid = dict(
        type='GeneralGridCoverage',
        srsName='EPSG:4326',  # TODO read from dataset
        axisLabels=list(ds.dims.keys()),
        axis=[_get_axis_properties(ds, dim) for dim in ds.dims]
    )
    domain_set = dict(
        type='DomainSet',
        generalGrid=grid
    )
    return domain_set


def _get_axis_properties(ds: xr.Dataset, dim: str):
    axis = ds.coords[dim]
    return dict(
        type='RegularAxis',
        axisLabel=dim,
        lowerBound=axis[0].item(),
        upperBound=axis[-1].item(),
        resolution=abs((axis[-1] - axis[0]).item() / len(axis)),
        uomLabel=_get_units(ds, dim)
    )


def _get_units(ds: xr.Dataset, dim: str):
    coord = ds.coords[dim]
    if hasattr(coord, 'attrs') and 'units' in coord.attrs:
        return coord.attrs['units']
    else:
        return 'degrees'


def _get_crs_from_dataset(ds: xr.Dataset):
