# xcube Dataset Specification

This document provides a technical specification of the protocol and 
format for *xcube datasets*, data cubes in the xcube sense. 

The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, 
“SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this 
document are to be interpreted as described in 
[RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Document Status

This is the latest version, which is still in development.

Version: 1.0, draft

Updated: 31.05.2018


## Motivation

For many users of Earth observation data, multivariate coregistration, 
extraction, comparison, and analysis of different data sources is 
difficult, while data is provided in various formats and at different 
spatio-temporal resolutions.

## High-level requirements

xcube datasets 

* SHALL be time series of gridded, geo-spatial, geo-physical variables.  
* SHALL use a common, equidistant, global or regional geo-spatial grid.
* SHALL shall be easy to read, write, process, generate.
* SHALL conform to the requirements of analysis ready data (ARD).
* SHALL be compatible with existing tools and APIs.
* SHALL conform to standards or common practices and follow a common 
  data model.
* SHALL be formatted as self-contained datasets.
* SHALL be "cloud ready", in the sense that subsets of the data can be
  accessed by individual URIs.

ARD links:

* http://ceos.org/ard/
* https://landsat.usgs.gov/ard
* https://medium.com/planet-stories/analysis-ready-data-defined-5694f6f48815
 

## xcube Dataset Schemas

### Basic Schema

* Attributes metadata convention 
  * SHALL be [CF](http://cfconventions.org/) >= 1.7 
  * SHOULD adhere to 
    [Attribute Convention for Data Discovery](http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery) 
* Dimensions: 
  * SHALL be at least `time`, `bnds`, and MAY be any others.
  * SHALL all be greater than zero, but `bnds` must always be two. 
* Temporal coordinate variables: 
  * SHALL provide time coordinates for given time index.
  * MAY be non-equidistant or equidistant. 
  * `time[time]` SHALL provide observation or average time of 
    *cell centers*. 
  * `time_bnds[time, bnds]` SHALL provide observation or integration 
    time of *cell boundaries*. 
  * Attributes: 
    * Temporal coordinate variables MUST have `units`, `standard_name`, 
      and any others.
    * `standard_name` MUST be `"time"`, `units` MUST have format 
      `"<deltatime> since <datetime>"`, where `datetime` must have 
      ISO-format. `calendar` may be given, if not, `"gregorian"` is 
      assumed.
* Spatial coordinate variables
  * SHALL provide spatial coordinates for given spatial index.
  * SHALL be equidistant in either angular or metric units 
* Cube variables: 
  * SHALL provide *cube cells* with the dimensions as index.
  * SHALL have shape 
    * `[time, ..., lat, lon]` (see WGS84 schema) or 
    * `[time, ..., y, x]` (see Generic schema) 
  * MAY have extra dimensions, e.g. `layer` (of the atmosphere), 
    `band` (of a spectrum).
  * SHALL specify the `units` metadata attribute.
  * SHOULD specify metadata attributes that are used to identify 
    missing values, namely `_FillValue` and / or `valid_min`, 
    `valid_max`, see notes in CF conventions on these attributes.
  * MAY specify metadata attributes that can be used to visualise the 
    data:
    * `color_bar_name`: Name of a predefined colour mapping. 
       The colour bar is applied between a minimum and a maximum value. 
    * `color_value_min`, `color_value_max`: Minimum and maximum value 
       for applying the colour bar. If not provided, minimum and maximum
       default to `valid_min`, `valid_max`. If neither are provided, 
       minimum and maximum default to `0` and `1`.

### WGS84 Schema (extends Basic)

* Dimensions:
  * SHALL be at least `time`, `lat`, `lon`, `bnds`, and MAY be any 
    others. 
* Spatial coordinate variables: 
  * SHALL use WGS84 (EPSG:4326) CRS.
  * SHALL have `lat[lat]` that provides observation or average latitude
    of *cell centers*
    with attributes: `standard_name="latitude"` `units="degrees_north"`.
  * SHALL have `lon[lon]` that provides observation or average longitude
    of *cell centers* with attributes: `standard_name="longitude"` and
    `units="degrees_east"`. 
  * SHOULD HAVE `lat_bnds[lat, bnds]`, `lon_bnds[lon, bnds]`: provide
    geodetic observation or integration coordinates of
    *cell boundaries*. 
* Cube variables: 
  * SHALL have shape `[time, ..., lat, lon]`. 

### Generic Schema (extends Basic)

* Dimensions: `time`, `y`, `x`, `bnds`, and any others. 
  * SHALL be at least `time`, `y`, `x`, `bnds`, and MAY be any others. 
* Spatial coordinate variables: 
  * Any spatial grid and CRS.
  * `y[y]`, `x[x]`: provide spatial observation or average coordinates
    of *cell centers*.
    * Attributes: `standard_name`, `units`, other units describe the 
      CRS / projections, see CF.
  * `y_bnds[y, bnds]`, `x_bnds[x, bnds]`: provide spatial observation
    or integration coordinates of *cell boundaries*.
  * MAY have `lat[y,x]`: latitude of *cell centers*. 
    *  Attributes: `standard_name="latitude"`, `units="degrees_north"`.
  * `lon[y,x]`: longitude of *cell centers*. 
    *  Attributes: `standard_name="longitude"`, `units="degrees_east"`.
* Cube variables: 
  * MUST have shape `[time, ..., y, x]`. 


## xcube EO Processing Levels

This section provides an attempt to characterize xcube datasets 
generated from Earth Observation (EO) data according to their 
processing levels as they are commonly used in EO data processing.

### Level-1C and Level-2C 

* Generated from Level-1A, -1B, -2A, -2B EO data.
* Spatially resampled to common grid
  * Typically resampled at original resolution.
  * May be down-sampled: aggregation/integration.
  * May be upsampled: interpolation.
* No temporal aggregation/integration.
* Temporally non-equidistant.

### Level-3

* Generated from Level-2C or -3 by temporal aggregation.
* No spatial processing.
* Temporally equidistant.
* Temporally integrated/aggregated.
