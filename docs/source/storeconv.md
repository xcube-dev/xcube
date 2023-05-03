# Common data store conventions

This document is a work in progress.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in [RFC 2119](https://tools.ietf.org/html/rfc2119).

Useful references related to this document include:

 - The [JSON Schema Specification](https://json-schema.org/specification.html)
   and the book [*Understanding JSON
   Schema*](https://json-schema.org/understanding-json-schema/)
 - [xcube Issue #330](https://github.com/dcs4cop/xcube/issues/330)
   (‘Establish common data store conventions’)
 - The existing xcube store plugins
   [xcube-sh](https://github.com/dcs4cop/xcube-sh/),
   [xcube-cci](https://github.com/dcs4cop/xcube-sh/), and
   [xcube-cds](https://github.com/dcs4cop/xcube-cds)
 - The
   [`xcube.util.jsonschema`](https://github.com/dcs4cop/xcube/blob/master/xcube/util/jsonschema.py) 
   source code

## Naming Identifiers

This section explains various identifiers used by the xcube data store 
framework and defines their format.

In the data store framework, identifiers are used to denote data sources, 
data stores, and data accessors.
Data store, data opener, and data writer identifiers are used to register the 
component as extension in a package's `plugin.py`. Identifiers MUST be 
unambiguous in the scope of the data store. 
They SHOULD be unambiguous across the entirety of data stores. 

There are no further restrictions for data source and data store identifiers.

A data accessor identifier MUST correspond to the following scheme:

`<data_type>:<format>:<storage>[:<version>]`

`<data_type>` identifies the in-memory data type to represent the data, 
e.g., `dataset` (or `xarray.Dataset`), `geodataframe` 
(or `geopandas.GeoDataFrame`).
`<format>` identifies the data format that may be accessed, 
e.g., `zarr`, `netcdf`, `geojson`.
`<storage>` identifies the kind of storage or data provision the 
accessor can access. Example values are `file` (the local file system), 
`s3` (AWS S3-compatible object storage), or `sentinelhub` 
(the Sentinel Hub API), or `cciodp` (the ESA CCI Open Data Portal API).
The `<version>` finally is an optional notifier 
about a data accessor's version. The version MUST follow the 
[Semantic Versioning](https://semver.org).

Examples for valid data accessors identifiers are:

* `dataset:netcdf:file`
* `dataset:zarr:sentinelhub`
* `geodataframe:geojson:file`
* `geodataframe:shapefile:cciodp:0.4.1`

## Open Parameters

This section aims to provide an overview of the interface defined by an xcube
data store or opener in its open parameters schema, and how this schema may be
used by a UI generator to automatically construct a user interface for a data
opener.

### Specification of open parameters

Every implementation of the `xcube.core.store.DataOpener` or
`xcube.core.store.DataStore` abstract base classes MUST implement the
`get_open_data_params_schema` method in order to provide a description of the
allowed arguments to `open_data` for each dataset supported by the
`DataOpener` or `DataStore`. The description is provided as a
`JsonObjectSchema` object corresponding to a [JSON
Schema](https://json-schema.org/). The intention is that this description 
should be full and detailed enough to allow the automatic construction of a 
user interface for access to the available datasets. Note that, under this 
system:

  1. Every dataset provided by an opener can support a different set of
     open parameters.

  2. The schema does not allow the representation of interdependencies between
     values of open parameters within a dataset. For instance, the following
     interdependencies between two open parameters *sensor_type* and
     *variables* would not be representable in an open parameters schema:
     
     *sensor_type*: A or B  
     *variables*: [temperature, humidity] for sensor type A;
                  [temperature, pressure] for sensor type B

To work around some of the restrictions of point (2) above, a dataset MAY be
presented by the opener as multiple "virtual" datasets with different
parameter schemas. For instance, the hypothetical dataset described above MAY 
be offered not as a single dataset `envdata` but as two datasets
`envdata:sensor-a` (with a fixed *sensor_type* of A) and `envdata:sensor-b`,
(with a fixed *sensor_type* of B), offering different sets of permitted
variables. 
Sometimes, the interdependencies between parameters are too complex to
be fully represented by splitting datasets in this manner. In these cases:

  1. The JSON Schema SHOULD describe the smallest possible superset of the
     allowed parameter combinations.

  2. The additional restrictions on parameter combinations MUST be clearly
     documented.

  3. If illegal parameter combinations are supplied, the opener MUST raise an
     exception with an informative error message, and the user interface
     SHOULD present this message clearly to the user.

### Common parameters

While an opener is free to define any open parameters for any of its datasets,
there are some common parameters which are likely to be used by the majority
of datasets. Furthermore, there are some parameters which are fundamental for 
the description of a dataset and therefore MUST be included in a schema 
(these parameters are denoted explicitly in the list below). In case that an 
opener does not support varying values of one of these parameters, a constant 
value must defined. This may be achieved by the JSON schema's `const` property 
or by an `enum` property value whose is a one-element array.

Any dataset requiring the specification of these parameters MUST
use the standard parameter names, syntax, and semantics defined below, in
order to keep the interface consistent. For instance, if a dataset allows a
time aggregation period to be specified, it MUST use the `time_period`
parameter with the format described below rather than some other alternative
name and/or format. Below, the parameters are described with their Python type
annotations.

 - `variable_names: List[str]`  
   A list of the identifiers of the requested variables. 
   This parameter MUST be included in an opener parameters schema.
   
 - `bbox: Union[str,Tuple[float, float, float, float]]`
   The bounding box for the requested data, in the order xmin, ymin, xmax, 
   ymax. Must be given in the units of the specified spatial coordinate 
   reference system `crs`. This parameter MUST be included in an opener 
   parameters schema. 
   
 - `crs: str`  
   The identifier for the spatial coordinate reference system of geographic 
   data.
   
 - `spatial_res: float`  
   The requested spatial resolution (x and y) of the returned data.
   Must be given in the units of the specified spatial coordinate reference 
   system `crs`.
   This parameter MUST be included in an opener parameters schema. 
   
 - `time_range: Tuple[Optional[str], Optional[str]]`  
   The requested time range for the data to be returned.
   The first member of the tuple is the start time; the second is the end 
   time. See section
   ‘[Date, time, and duration specifications](#sec-datespec)’.
   This parameter MUST be included in an opener parameters schema.
   If a date without a time is given as the start time,
   it is interpeted as 00:00 on the specified date.
   If a date without a time is given as the end time,
   it is interpreted as 24:00 on the specified date
   (identical with 00:00 on the date following the specified date).
   If the end time is specified as `None`,
   it is interpreted as the current time.
   
 - `time_period: str`  
   The requested temporal aggregation period for the data. See section
   ‘[Date, time, and duration specifications](#sec-datespec)’.
   This parameter MUST be included in an opener parameters schema.
   
 - `force_cube: bool`  
   Whether to return results as a [specification-compliant
   xcube](https://github.com/dcs4cop/xcube/blob/master/docs/source/cubespec.md).
   If a store supports this parameter and if a dataset is opened with this
   parameter set to `True`, the store MUST return a specification-compliant
   xcube dataset. If this parameter is not supported or if a dataset is opened 
   with this parameter set to `False`, the caller MUST NOT assume that the 
   returned data conform to the xcube specification.

### Semantics of list-valued parameters

The `variables` parameter takes as its value a list, with no duplicated members
and the values of its members drawn from a predefined set. The values of this
parameter, and other parameters whose values also follow such a format, are
interpreted by xcube as a *restriction*, much like a bounding box or time
range. That is:

 - By default (if the parameter is omitted or if a `None` value is supplied
   for it), *all* the possible member values MUST be included in the list. In
   the case of `variables`, this will result in a dataset containing all the
   available variables.
 - If a list containing *some* of the possible members is given, a dataset
   corresponding to those members only MUST be returned. In the case of
   `variables`, this will result in a dataset containing only the requested
   variables.
 - A special case of the above: if an empty list is supplied, a dataset
   containing *no data* MUST be returned -- but with the requested spatial and
   temporal dimensions.

### <a id="sec-datespec"></a>Date, time, and duration specifications

In the common parameter `time_range`, times can be specified using the
standard JSON Schema formats `date-time` or `date`. Any additional time or
date parameters supported by an xcube opener dataset SHOULD also use these
formats, unless there is some good reason to prefer a different format.

The formats are described in the [JSON Schema Validation 2019
draft](https://json-schema.org/draft/2019-09/json-schema-validation.html#rfc.section.7.3.1),
which adopts definitions from [RFC 3339 Section
5.6](https://tools.ietf.org/html/rfc3339#section-5.6). The JSON Schema
`date-time` format corresponds to RFC 3339's `date-time` production, and JSON
Schema's `date` format to RFC 3339's `full-date` production. These formats are
subsets of the widely adopted [ISO
8601](https://en.wikipedia.org/wiki/ISO_8601) format.

The `date` format corresponds to the pattern `YYYY-MM-DD` (four-digit year –
month – day), for example `1995-08-20`. The `date-time` format consists of a
date (in the `date` format), a time (in `HH:MM:SS` format), and timezone (`Z`
for UTC, or `+HH:MM` or `-HH:MM` format).  The date and time are separated by
the letter `T`. Examples of `date-time` format include `1961-03-23T12:22:45Z`
and `2018-04-01T21:12:00+08:00`. Fractions of a second MAY also be included,
but are unlikely to be relevant for xcube openers.

The format for durations, as used for aggregation period, does **not** conform
to the syntax defined for this purpose in the ISO 8601 standard (which is also
quoted as Appendix A of RFC 3339). Instead, the required format is a small
subset of the [pandas time series frequency
syntax](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases), 
defined by the following regular expression:


```
^([1-9][0-9]*)?[HDWMY]$
```

That is: an optional positive integer followed by one of the letters H (hour),
D (day), W (week), M (month), and Y (year). The letter specifies the time unit
and the integer specifies the number of units. If the integer is omitted, 1 is
assumed.

### Time limits: an extension to the JSON Schema

JSON Schema itself does not offer a way to impose time limits on a string
schema with the `date` or `date-time` format. This is a problem for xcube
generator UI creation, since it might be reasonably expected that a UI will
show and enforce such limits. The xcube opener API therefore defines an
unofficial extension to the JSON string schema: a `JsonStringSchema` object
(as returned as part of a `JsonSchema` by a call to
`get_open_data_params_schema`) MAY, if it has a `format` property with a value
of `date` or `date-time`, also have one or both of the properties
`min_datetime` and `max_datetime`. These properties must also conform to the
`date` or `date-time` format.  xcube provides a dedicated `JsonDatetimeSchema`
for this purpose. Internally, it extends `JsonStringSchema` by adding the
required properties to the JSON string schema.

### Generating a UI from a schema

With the addition of the time limits extension described above, the JSON
Schema returned by `get_open_data_params_schema` is expected to be extensive
and detailed enough to fully describe a UI for cube generation.

#### Order of properties in a schema

Sub-elements of a `JsonObjectSchema` are passed to the constructor using the
`properties` parameter with type signature `Mapping[str, JsonSchema]`. Openers
SHOULD provide an ordered mapping as the value of `properties`, with the
elements placed in an order suitable for presentation in a UI, and UI
generators SHOULD lay out the UI in the provided order, with the exception of
the common parameters discussed below. Note that the CPython `dict` object
preserves the insertion order of its elements as of Python 3.6, and that this
behaviour is officially guaranteed as of Python 3.7, so additional classes
like `OrderedDict` are no longer necessary to fulfil this requirement.

#### Special handling of common parameters

Any of the common parameters listed above SHOULD, if present, be recognized
and handled specially. They SHOULD be presented in a consistent position
(e.g. at the top of the page for a web GUI), in a consistent order, and with
user-friendly labels and tooltips even if the `title` and `description`
annotations (see below) are absent. The UI generator MAY provide special
representations for these parameters, for instance an interactive map for the
`bbox` parameter.

An opener MAY provide `title`, `description`, and/or `examples` annotations
for any of the common parameters, and a UI generator MAY choose to use any of
these to supplement or modify its standard presentation of the common
parameters.

#### Schema annotations (title, description, examples, and default)

For JSON Schemas describing parameters other than the common parameters, an
opener SHOULD provide the `title` and `description` annotations. A UI
generator SHOULD make use of these annotations, for example by taking the
label for a UI control from `title` and the tooltip from `description`. The
opener and UI generator MAY additionally make use of the `examples` annotation
to record and display example values for a parameter. If a sensible default
value can be envisaged, the opener SHOULD record this default as the value of
the `default` annotation and the UI generator SHOULD set the default value in
the UI accordingly. If the `title` annotation is absent, the UI generator
SHOULD use the key corresponding to the parameter's schema in the parent
schema as a fallback.

#### Generalized conversion of parameter schemas

For parameters other than the common parameters, the UI can be generated
automatically from the schema structure. In the case of a GUI, a one-to-one
conversion of values of JSON Schema properties into GUI elements will
generally be fairly straightforward. For instance:

 - A schema of type `boolean` can be represented as a checkbox.

 - A schema of type `string` without restrictions on allowed items can be
   represented as an editable text field.

 - A schema of type `string` with an `enum` keyword giving a list of
   allowed values can be represented as a drop-down menu.

 - A schema of type `string` with the keyword setting `"format": "date"` can
   be represented as a specialized date selector.

 - A schema of type `array` with the keyword setting `"uniqueItems": true` and
   an `items` keyword giving a fixed list of allowed values can be represented
   as a list of checkboxes.
