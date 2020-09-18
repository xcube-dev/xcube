# The xcube store plugin API

## Opener parameters

Every implementation of an xcube `DataOpener` or `DataStore` must implement
the `get_open_data_params_schema` method in order to provide a description
of the allowed arguments to `open_data` for each dataset supported by the
`DataOpener` or `DataStore`. The description is provided as a `JsonObjectSchema`
object corresponding to a [JSON Schema](https://json-schema.org/). The
intention is that this description should be full and detailed enough to allow
the automatic construction of a user interface for access to the available
datasets. Note that, under this system:

  1. Every dataset which a store provides can support a different set of
     opener parameters.

  2. The schema does not allow the representation of interdependencies
     between values of opener parameters within a dataset. For
     instance, the following cannot be represented in an open
     parameters schema:
     
     *sensor_type*: A or B  
     *variables*: [temperature, humidity] for sensor type A;
                  [temperature, pressure] for sensor type B

To work around some of the restrictions of point (2) above, a dataset may be
presented by the store as multiple "virtual" datasets with difference 
parameter schemas. For instance, the hypothetical dataset described above
could be offered not as a single dataset `envdata` but as two datasets
`envdata:sensor-a` and `envdata:sensor-b`, with different sets of permitted
variables.

Sometimes, the interdependencies between parameters are too complex to
be represented fully by splitting datasets in this way. In these cases:

  1. The JSON Schema should describe the smallest possible superset of the
     allowed parameter combinations.

  2. The additional restrictions on parameter combinations should be
     clearly documented.

  3. If illegal parameter combinations are supplied, the store plugin should
     raise an exception with an informative error message, and the
     user interface should present this message clearly to the user.

## Common parameters

While a store is free to define any opener parameters for any of its
datasets, there is are some common parameters which are likely to be
used by the majority of datasets. Any dataset requiring the specification
of these parameters should use the standard parameter names, syntax, and
semantics defined below, in order to keep the interface consistent.
For instance, if a dataset allows a time aggregation period to be specified,
it should use the `time_period` parameter with the format described below
rather than some other parameter name and/or format.

```
variable_names: List[str]
bbox:Union[str,Tuple[float, float, float, float]]
crs:str
spatial_res: float
time_range: Tuple[Optional[str]
Optional[str]]
time_period: str
```

## Date, time, and duration specifications

In the common parameter `time_range`, times can be specified using the
standard JSON Schema formats `date-time` or `date`. It is suggested
that any additional time or date parameters supported by an xcube
store dataset also use these formats, unless there is some good reason
to prefer a different format.

The formats are described in the [JSON Schema Validation 2019
draft](https://json-schema.org/draft/2019-09/json-schema-validation.html#rfc.section.7.3.1),
which in turn adopts definitions from [RFC 3339 Section
5.6](https://tools.ietf.org/html/rfc3339#section-5.6). The JSON Schema
`date-time` format corresponds to RFC 3339's `date-time` production,
and JSON Schema's `date` format to RFC 3339's `full-date`
production. These formats are subsets of the widely adopted [ISO
8601](https://en.wikipedia.org/wiki/ISO_8601) format.

The `date` format corresponds to the pattern `YYYY-MM-DD` (four-digit
year – month – day), for example `1995-08-20`. The `date-time` format
consists of a date (in the `date` format), a time (in `HH:MM:SS`
format), and timezone (`Z` for UTC, or `+HH:MM` or `-HH:MM` format).
The date and time are separated by the letter `T`. Examples of `date-time`
format include `1961-03-23T12:22:45Z` and `2018-04-01T21:12:00+08:00`.
(Fractions of a second may also be included, but are unlikely to be
relevant for xcube stores.)

## Time limits: an extension to the JSON Schema

TODO

## Generating a UI from a schema

TODO
