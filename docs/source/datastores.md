[xcube Dataset Convention]: ./cubespec.md
[xarray.Dataset]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
[ESA Climate Data Centre]: https://climate.esa.int/en/odp/
[Sentinel Hub]: https://www.sentinel-hub.com/
[xcube.core.store]: https://github.com/dcs4cop/xcube/tree/master/xcube/core/store
# Data stores

## Data (cube) sources

In xcube, data cubes are raster datasets that are basically a collection 
of N-dimensional geo-physical variables represented by
[xarray.Dataset] Python objects (see also [xcube Dataset Convention]). 
Data cubes may be provided by a variety of sources and may be stored using 
different data formats. In the simplest case you have a NetCDF file or a 
Zarr directory in your local filesystem that already represents a data cube. 
Data cubes may also be stored on AWS S3 or Google Cloud Storage using the 
Zarr format. Sometimes a set of NetCDF or GeoTIFF files in some storage 
must first be concatenated to form a data cube. In other cases, data cubes 
can be generated on-the-fly by suitable requests to some cloud-hosted data 
API such as the [ESA Climate Data Centre] or [Sentinel Hub].

## Data store framework

The _xcube data store framework_ provides a simple and consistent Python 
interface that is used to open [xarray.Dataset] objects from _data stores_ 
which abstract away the individual data sources, protocols, formats and hides 
involved data processing steps. The following two lines open a data cube
from the [ESA Climate Data Centre] comprising the essential climate variable 
Sea Surface Temperature (SST):

```python
store = new_data_store("cciodp")
dataset = store.open_data("esacci.SST.day.L4.SSTdepth.multi-sensor.multi-platform.OSTIA.1-1.r1")
```

Data stores can also be writable. All read-only or writable data stores 
share the same functional interface, however their configuration may change. 
Depending on what is offered by a given data store, also the parameters 
passed to the `open_data()` method may change. 

Data stores are xcube extensions; data stores can be added by xcube plugins.

Create lazy data cube views (xarray.Dataset with Dask arrays)

Data stores can serve cubes, multi-resolution cubes, and vector data

The xcube data store framework is implemented in the [xcube.core.store] 
package, see also its [public API reference]().


## Data store implementations

## Developing a new data store
