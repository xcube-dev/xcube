xcube Multi-Resolution Datasets
===============================

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
It is planned to also support Cloud Optimized GeoTIFF (COG) as format 
for multi-resolution datasets in xcube.

The xcube Levels Format
-----------------------

The xcube Levels format is basically a single top-level directory. 
The filename extension of that directory should be `.levels` 
by convention. The directory entries are Zarr datasets   

1. that are representations of regular xarray datasets named after 
   their zero-based level index, `{level}.zarr`;
2. that comply with the xcube dataset convention.

TODO (forman): link to xcube dataset convention

The following is a multi-resolution dataset with three levels:

    - test_pyramid.levels/
        - 0.zarr/
        - 1.zarr/
        - 2.zarr/

An important use case is generating image pyramids from existing large 
datasets without the need to create a copy of level zero.

To support this, the level zero dataset may be a link to an existing 
Zarr dataset. The filename is then `0.link` rather than `0.zarr`. 
The link file contains the path to the actual Zarr dataset 
to be used as level zero as a plain text string. It may be an absolute 
path or a path relative to the top-level dataset.

    - test_pyramid.levels/
        - 0.link    # --> link to actual level zero dataset
        - 1.zarr/
        - 2.zarr/

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
* Make top-level directory a Zarr group (`.zgroup`)
  and encode level metadata (e.g. `num_levels` and level links) in `.zattrs`, or 
  even better `.zlevels`?
* Allow a Zarr levels sub-group of the level zero Zarr dataset.
  It would contain all levels without level zero, hence avoiding the need to 
  link level zeros. 

To do
-----

* Currently, the FS data stores treat relative link paths as relative
  to the data store's `root`. See https://github.com/dcs4cop/xcube/pull/637

