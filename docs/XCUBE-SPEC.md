# xcube Specification version 1


--------------------------------------

This document provides a technical specification of the protocol and format used 
for xcube data cubes. The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”,
 “SHALL NOT”, “SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in 
 this document are to be interpreted as described in 
 [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Status

This is the latest version, which is still in development.

Version: 1.0, draft

Updated: 31.05.2018


## Motivation

For many users of Earth observation data, multivariate coregistration, 
extraction, comparison, and analysis of different data sources is difficult,
when data is provided in various formats and at different spatio-temporal 
resolutions.


## High-level requirements


xcube data 

* SHALL be time series of gridded, geo-spatial, geo-physical variables  
* SHALL use a common, equidistant, global or regional geo-spatial grid
* SHALL shall be easy to read, write, process, generate
* SHALL conform to the requirements of analysis ready data (ARD)
* SHALL be compatible with existing tools and APIs
* SHALL conform to standards or common practices and follow a common data model
* SHALL be formatted as self-contained datasets
* SHALL be Cloud ready (TBD...)

ARD links:

* http://ceos.org/ard/
* https://landsat.usgs.gov/ard
* https://medium.com/planet-stories/analysis-ready-data-defined-5694f6f48815
 

## xcube Schemas

### Basic Schema

* Attributes metadata convention 
  * SHALL be CF >= 1.6 
  * SHOULD adhere to THREDDS data server catalogue metadata 
* Dimensions: 
  * SHALL be at least `time`, `bnds`, and MAY be any others.
  * SHALL all be greater than zero, but `bnds` must always be two. 
* Temporal coordinate variables: 
  * SHALL provide time coordinates for given time index.
  * MAY be non-equidistant or equidistant. 
  * `time[time]` SHALL provide observation or average time of *cell centers*. 
  * `time_bnds[time, bnds]` SHALL provide observation or integration time of *cell boundaries*. 
  * Attributes: 
    * Temporal coordinate variables MUST have `units`, `standard_name`, and any others.
    * `standard_name` MUST be `"time"`, `units` MUST have format `"<deltatime> since <datetime>"` 
       where `datetime` must have ISO-format. `calendar` may be given, if not,
      `"gregorian"` is assumed.
* Spatial coordinate variables
  * SHALL provide spatial coordinates for given spatial index.
  * SHALL be equidistant in either angular or metric units 
* Cube variables: 
  * SHALL provide *cube cells* with the dimensions as index.
  * SHALL have shape 
    * `[time, ..., lat, lon]` (see WGS84 schema) or 
    * `[time, ..., y, x]` (see Generic schema) 
  * MAY have extra dimensions, e.g. `layer` (of the atmosphere), `band` (of a spectrum).


### WGS84 Schema (extends Basic)

* Dimensions:
  * SHALL be at least `time`, `lat`, `lon`, `bnds`, and MAY be any others. 
* Spatial coordinate variables: 
  * SHALL use WGS84 (EPSG:4326) CRS.
  * SHALL have `lat[lat]` that provides observation or average latitude of *cell centers*
    with attributes: `standard_name="latitude"` `units="degrees north"`.
  * SHALL have `lon[lon]` that provides observation or average longitude of *cell centers* 
    with attributes: `standard_name="longitude"` and `units="degrees east"` 
  * SHOULD HAVE `lat_bnds[lat, bnds]`, `lon_bnds[lon, bnds]`: provide geodetic observation or integration coordinates of *cell boundaries*. 
* Cube variables: 
  * SHALL have shape `[time, ..., lat, lon]`. 

### Generic Schema (extends Basic)

* Dimensions: `time`, `y`, `x`, `bnds`, and any others. 
  * SHALL be at least `time`, `y`, `x`, `bnds`, and MAY be any others. 
* Spatial coordinate variables: 
  * Any spatial grid and CRS.
  * `y[y]`, `x[x]`: provide spatial observation or average coordinates of *cell centers*.
    *  Attributes: `standard_name`, `units`, other units describe the CRS / projections, see CF.
  * `y_bnds[y, bnds]`, `x_bnds[x, bnds]`: provide spatial observation or integration coordinates of *cell boundaries*.
  * MAY have `lat[y,x]`: latitude of *cell centers*. 
    *  Attributes: `standard_name="latitude"`, `units="degrees north"`.
  * `lon[y,x]`: longitude of *cell centers*. 
    *  Attributes: `standard_name="longitude"`, `units="degrees east"`.
* Cube variables: 
  * MUST have shape `[time, ..., y, x]`. 



## xcube Processing Levels


### Level-2C 

* Generated from Level-2 Earth Observation data
* Spatially resampled to common grid
  * Typically resampled at original resolution
  * May be down-sampled: aggregation/integration
  * May be upsampled: interpolation
* No temporal aggregation/integration
* Temporally non-equidistant

### Level-3

* Generated from Level-2C xcubes
* No spatial processing
* Temporally equidistant
* Temporally integrated/aggregated

## Further ideas 

* Multi-resolution xcubes: embed cube pyramids in xcube at different spatial resolutions.
  Usages: fast image tile server, multi-resolution analysis (e.g. feature detection, algal blooms)
* Multi-chunks xcubes: embed differently chunked cubes in an xcube.
  Usages: fast time-series analyses.   

Provide a Python API based on xarray addressing the following operations

* Merge
* Combine
* Fill gaps
* Match-up
* Extract where
* Subset

## Implementation Hints

For multiple regional cubes that "belong together" (e.g. one project)
use common resolutions and regions that snap on a Fixed Earth grid, which has been
been defined with respect to ideal tile / chunk sizes. 

Chunking of data has a very high impact on processing performance:

* If xcubes are served by a tile map server, tile sizes shall be aligned with chunk sizes
* xcube spatial image size shall be integer divisible by chunk sizes 


The data cubes in the DCS4COP project should be generated for predefined 
regions and predefined spatial resolutions. A cube for a given region may be 
generated for multiple spatial resolutions.

There may be a scenario in which all the region cubes for will be used together. Then,
in order to avoid another spatial resampling to a common grid, the grids of all cubes should be
defined on a fixed Earth grid.

This means a cube's region coordinates are taken from the corresponding cells 
of a fixed Earth grid defined for a given spatial resolution.

Therefore the `num_levels` spatial resolutions in which data cubes are produced shall be defined such that
the fixed Earth's grid sizes multiplied by the cell size `delta` is close to 180 degrees.
In addition we want the actual data cubes sizes to be integer multiples of an appropriate internal
chunk (tile) size used to compress and store the data at different resolution levels `i`:

    |grid_size[i] * delta[i] - 180| < eps
    num_chunks[i] = num_chunks[0] * power(2, i)
    grid_size[i] = num_chunks[i] * chunk_size

with

    0 <= i < num_levels
    num_chunks[0] := number of chunks at lowest spatial resolution i in both spatial dimensions
    chunk_size := size of a chunk in both spatial dimensions



