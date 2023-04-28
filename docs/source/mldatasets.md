xcube Multi-Resolution Datasets
===============================

Version 1.0 Draft, 2023-04-28

Definition
----------

A xcube _multi-resolution dataset_ refers to an N-D [image 
pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing)) 
where an _image_ refers to a 2-D dataset with two spatial dimensions
in some horizontal coordinate system.

A multi-resolution dataset comprises a fixed number of
_levels_, which are regular datasets covering the same spatial area at 
different resolutions. Level zero represents the original resolution 
`res(L=0)`, higher level resolutions decrease by a factor of two: 
`res(L) = res(0) / 2^L`.


Implementation in xcube
-----------------------

In xcube, multi-resolution datasets are represented by the abstract class
`xcube.core.mldataset.MultiLevelDataset`. The xcube data store framework
refers to this datatype using the alias `mldataset`. The corresponding
default data format is the xcube _Levels_ format, named `levels`.
xcube also supports the Cloud Optimized GeoTIFF (COG) format 
for reading multi-resolution datasets.

The xcube Levels Format
-----------------------

The xcube Levels format is basically a single top-level directory. 
The filename extension of that directory should be `.levels` 
by convention. The directory entries are Zarr datasets   

1. that are representations of regular xarray datasets named after 
   their zero-based level index, `{level}.zarr`;
2. that comply with the [xcube Dataset Convention](./cubespec.md).

The following is a multi-resolution dataset with three levels:

```text
- test_pyramid.levels/
    - 0.zarr/
    - 1.zarr/
    - 2.zarr/
```

An important use case is generating image pyramids from existing large 
datasets without the need to create a copy of level zero.

To support this, the level zero dataset may be a link to an existing 
Zarr dataset. The filename is then `0.link` rather than `0.zarr`. 
The link file contains the path to the actual Zarr dataset 
to be used as level zero as a plain text string. It may be an absolute 
path or a path relative to the top-level dataset.

```text
- test_pyramid.levels/
    - 0.link    # --> link to actual level zero dataset
    - 1.zarr/
    - 2.zarr/
```

Starting with xcube 0.13.1, an additional, optional file `.zlevels` 
has been made part of the levels format:

```text
- test_pyramid.levels/
    - .zlevels
    - 0.zarr/
    - 1.zarr/
    - 2.zarr/
```

If present, it is a text file comprising a JSON object with the following 
properties:

| Name               | Type                 | Description                                                   |
|--------------------|----------------------|---------------------------------------------------------------|
| `version`          | `"1.0"`              | Levels format version.                                        |
| `num_levels`       | integer              | Number of levels in this dataset                              |
| `use_saved_levels` | boolean              | If a next level shall be computed from the predecessor level. |
| `tile_size`        | \[integer, integer\] | Tile size width and height in pixels.                         |
| `agg_methods`      | object               | Mapping from variable name to aggregation method.             |

Only `version` and `num_levels` are required.

The properties of the `agg_methods` objects are the names of data variables.
The values are aggregation methods. Valid values are

| Value    | Description                                                  |
|----------|--------------------------------------------------------------|
| `first`  | Select the first pixel at (0,0) of a window of N x N pixels. | 
| `min`    | Minimum value of a window of N x N pixels.                   | 
| `max`    | Minimum value of a window of N x N pixels.                   | 
| `mean`   | Mean value of a window of N x N pixels.                      | 
| `median` | Median value of a window of N x N pixels.                    | 

The following is an example of the `.zlevels` file for a dataset with the 
data variables `CHL` (chlorophyll) if type `float32` and a variable 
`qflags` of type `uint16`:

```json
{
  "version": "1.0",
  "num_levels": 8,
  "use_saved_levels": true,
  "tile_size": [2048, 2048],
  "agg_methods": {
    "CHL": "median",
    "qflags": "first"
  }
}
```

---

**xcube implementation note**: 
When writing datasets as multi-level datasets and the `agg_methods` 
parameter is missing, or a data variable's name is not contained in
given `agg_methods` then `first` is used for variables that have 
an integer data type and `median` for a floating point data type.
In xcube Server, when opening datasets and converting them into 
multi-level datasets on-the-fly, `agg_methods` is `first` for all 
data variables for best performance. 

---


Related reads
-------------

* [WIP: Multiscale use-case](https://github.com/zarr-developers/zarr-specs/issues/23)
  in zarr-developers / zarr-specs on GitHub.
* [Multiscale convention](https://github.com/zarr-developers/zarr-specs/issues/125)
  in zarr-developers / zarr-specs on GitHub.
* [Package ndpyramid](https://github.com/carbonplan/ndpyramid)


To be discussed
---------------

* Allow links for all levels?
* Do not write `0.link` file. Instead, provide in `.zlevels` where to find 
  each level.
* No longer use `.zarr` extension for levels. Just use the index as name.
* Make top-level directory a Zarr group (`.zgroup`), so the multi-level 
  dataset can be opened as a group using the `zarr` package.


