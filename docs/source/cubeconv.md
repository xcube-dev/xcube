# xcube Dataset Convention

This document describes a convention for *xcube datasets*, which are data cubes 
in the xcube sense. Any dataset can be considered a data cube as long as at 
least a subset of its data variables are cube-like, i.e., meet the requirements 
listed in this document. 

The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, 
“SHOULD”, “SHOULD NOT”, “RECOMMENDED”, “MAY”, and “OPTIONAL” in this 
document are to be interpreted as described in 
[RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

## Document Status

This is the latest version, which is still in development.

Version: 1.0, draft

Updated: 21.07.2021


## Motivation

For many users of Earth observation data, common operations such as 
multivariate co-registration, extraction, comparison, and analysis of different 
data sources are difficult, while data is provided in various formats and at 
different spatio-temporal resolutions.

## High-level requirements

xcube datasets 

* SHALL be time series of gridded, geo-spatial, geo-physical variables.  
* SHALL use a common, equidistant, global or regional geo-spatial grid.
* SHALL be easy to read, write, process, generate.
* SHALL conform to the requirements of analysis ready data (ARD).
* SHALL be compatible with existing tools and APIs.
* SHALL conform to standards or common practices and follow a common 
  data model.
* SHALL be formatted as self-contained datasets.
* SHALL be "cloud ready", in the sense that subsets of the data can be
  accessed by individual URIs.

ARD links:

* http://ceos.org/ard/
* https://www.usgs.gov/core-science-systems/nli/landsat/us-landsat-analysis-ready-data
* https://medium.com/planet-stories/analysis-ready-data-defined-5694f6f48815
 

## xcube Dataset Schemas

### Basic Schema

* Attributes 
  * SHALL be [CF](http://cfconventions.org/) >= 1.7 
  * SHOULD adhere to 
    [Attribute Convention for Data Discovery](http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery) 
* Dimensions: 
  * SHALL all be greater than zero.
  + SHALL include two spatial dimensions  
  * SHOULD include a dimension `time`
  * SHOULD include a dimension `bnds` of size 2 that may be used by bounding 
    coordinate variables
* Coordinate Variables
  * SHALL contain labels for a dimension
  * SHOULD be 1-dimensional
  * MAY be 2-dimensional if, e.g., they are bound coordinate variables (see 
    below) or they carry `latitude`/`longitude` values in case of
  * 1-dimensional coordinate variables SHOULD be named like the dimension they 
    describe 
  * For each dimension of a data variable, a coordinate variable MUST exist
* Temporal coordinate variables: 
  * SHALL provide time coordinates for a given time index.
  * MAY be non-equidistant or equidistant.
  * SHOULD be named `time`    
  * One variable value SHALL provide observation or average time of 
    *cell centers*.
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
  * Different spatial coordinate variables MAY have different spatial 
    resolutions 
* Bound coordinate variables
  * SHOULD be included for any spatial or temporal coordinate variable
  * SHALL consist of two dimensions: The one of the respective coordinate 
    variable and another one of length 2, that SHOULD be named `bnds`
  * SHOULD be named `<dim_name>_bnds`
  * `<bound_var>[<coord_dim>, 0]` SHALL provide the *lower cell boundary*,
    `<bound_var>[<coord_dim>, 1]` SHALL provide the *upper cell boundary*
* Data variables: 
  * MAY have any dimensionality, including no dimensions at all.
  * SHALL have the spatial dimensions at the innermost position in case it has 
    spatial dimensions (e.g., `[..., y, x]`)
  * SHALL have its time dimension at the outermost position in case it has a
    time dimension (e.g., `[time, ...]`)
  * MAY have extra dimensions, e.g. `layer` (of the atmosphere) or 
    `band` (of a spectrum). These extra dimensions MUST be positioned between
    the time and the spatial coordinates
  * SHALL provide *cube cells* with the dimensions as index.
  * SHOULD specify the `units` metadata attribute.
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
  * SHALL include two spatial dimensions, which SHOULD be named `lat` and `lon`
* Spatial coordinate variables: 
  * SHALL use WGS84 (EPSG:4326) CRS.
  * One entry of the variable describing the latitude SHALL provide the 
    observation or average latitude of *cell centers*. It SHOULD have the 
    attributes: `standard_name="latitude"` `units="degrees_north"`.
  * One entry of the variable describing the longitude SHALL provide the 
    observation or average longitude of *cell centers*. It SHOULD have the 
    attributes: `standard_name="longitude"` `units="degrees_east"`.

### Generic Schema (extends Basic)

* Dimensions:
  * SHALL include two spatial dimensions, which SHOULD be named `y` and `x`
* Spatial coordinate variables: 
  * MAY use any spatial grid and CRS.
  * SHOULD have attributes `standard_name`, `units`
  * MAY have `lat[<y_dim_name>,<x_dim_name>]`: latitude of *cell centers*. 
    *  Attributes: `standard_name="latitude"`, `units="degrees_north"`.
  * MAY have `lon[<y_dim_name>,<x_dim_name>]`: longitude of *cell centers*. 
    *  Attributes: `standard_name="longitude"`, `units="degrees_east"`.
* Grid Mapping variable:
  * SHALL be included in case the CRS is not WGS84.
  * SHALL not carry any data, therefore it MAY be of any type
  * SHOULD be named `crs`  
  * MUST have attributes that describe a CF Grid Mapping v1.8 (see 
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#grid-mappings-and-projections 
    ). This means that there MUST either be 
      * an attribute `crs_wkt` that desribes a CRS in WKT format
      * an attribute `spatial_ref` (e.g., an EPSG code)
      * an attribute `grid_mapping_name`. If this is given, more attributes 
        MAY be required, depending on the grid mapping.



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
