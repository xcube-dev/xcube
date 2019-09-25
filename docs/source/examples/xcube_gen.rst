.. _xcube repository: https://github.com/dcs4cop/xcube/tree/master/examples/gen/data
.. _Copernicus Marine Environment Monitoring Service: http://marine.copernicus.eu/
.. _Product User Manual: http://resources.marine.copernicus.eu/documents/PUM/CMEMS-SST-PUM-010-001.pdf
.. _configuration file: https://github.com/dcs4cop/xcube/tree/master/examples/gen/config_files/xcube_sst_demo_config.yml
.. _configuration file which takes Sentinel-3 Ocean and Land Colour Instrument (OLCI): https://github.com/dcs4cop/xcube/tree/master/examples/gen/config_files/xcube_olci_demo_config.yml

.. warning:: This chapter is a work in progress and currently less than a draft.

===========================
Generating an xcube dataset
===========================

In the following example a tiny demo xcube dataset is generated.


Analysed Sea Surface Temperature over the Global Ocean
========================================================

Input data for this example is located in the `xcube repository`_.
The input files contain analysed sea surface temperature and sea surface temperature anomaly over the global ocean
and are provided by `Copernicus Marine Environment Monitoring Service`_.
The data is described in a dedicated `Product User Manual`_.

Before starting the example, you need to activate the xcube environment:

::

    $ conda activate xcube

If you want to take a look at the input data you can use :doc:`cli/xcube dump` to print out the metadata of a selected input file:

::

    $ xcube dump examples/gen/data/20170605120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc

::

        <xarray.Dataset>
        Dimensions:       (lat: 720, lon: 1440, time: 1)
        Coordinates:
          * lat           (lat) float32 -89.875 -89.625 -89.375 ... 89.375 89.625 89.875
          * lon           (lon) float32 0.125 0.375 0.625 ... 359.375 359.625 359.875
          * time          (time) object 2017-06-05 12:00:00
        Data variables:
            sst_anomaly   (time, lat, lon) float32 ...
            analysed_sst  (time, lat, lon) float32 ...
        Attributes:
            Conventions:                CF-1.4
            title:                      Global SST & Sea Ice Anomaly, L4 OSTIA, 0.25 ...
            summary:                    A merged, multi-sensor L4 Foundation SST anom...
            references:                 Donlon, C.J., Martin, M., Stark, J.D., Robert...
            institution:                UKMO
            history:                    Created from sst:temperature regridded with a...
            comment:                    WARNING Some applications are unable to prope...
            license:                    These data are available free of charge under...
            id:                         UKMO-L4LRfnd_GLOB-OSTIAanom
            naming_authority:           org.ghrsst
            product_version:            2.0
            uuid:                       5c1665b7-06e8-499d-a281-857dcbfd07e2
            gds_version_id:             2.0
            netcdf_version_id:          3.6
            date_created:               20170606T061737Z
            start_time:                 20170605T000000Z
            time_coverage_start:        20170605T000000Z
            stop_time:                  20170606T000000Z
            time_coverage_end:          20170606T000000Z
            file_quality_level:         3
            source:                     UKMO-L4HRfnd-GLOB-OSTIA
            platform:                   Aqua, Envisat, NOAA-18, NOAA-19, MetOpA, MSG1...
            sensor:                     AATSR, AMSR, AVHRR, AVHRR_GAC, SEVIRI, TMI
            metadata_conventions:       Unidata Observation Dataset v1.0
            metadata_link:              http://data.nodc.noaa.gov/NESDIS_DataCenters/...
            keywords:                   Oceans > Ocean Temperature > Sea Surface Temp...
            keywords_vocabulary:        NASA Global Change Master Directory (GCMD) Sc...
            standard_name_vocabulary:   NetCDF Climate and Forecast (CF) Metadata Con...
            westernmost_longitude:      0.0
            easternmost_longitude:      360.0
            southernmost_latitude:      -90.0
            northernmost_latitude:      90.0
            spatial_resolution:         0.25 degree
            geospatial_lat_units:       degrees_north
            geospatial_lat_resolution:  0.25 degree
            geospatial_lon_units:       degrees_east
            geospatial_lon_resolution:  0.25 degree
            acknowledgment:             Please acknowledge the use of these data with...
            creator_name:               Met Office as part of CMEMS
            creator_email:              servicedesk.cmems@mercator-ocean.eu
            creator_url:                http://marine.copernicus.eu/
            project:                    Group for High Resolution Sea Surface Tempera...
            publisher_name:             GHRSST Project Office
            publisher_url:              http://www.ghrsst.org
            publisher_email:            ghrsst-po@nceo.ac.uk
            processing_level:           L4
            cdm_data_type:              grid


Below an example xcube dataset will be created, which will contain the variable analysed_sst.
The metadata for a specific variable can be viewed by:

::

    $ xcube dump examples/gen/data/20170605120000-UKMO-L4_GHRSST-SSTfnd-OSTIAanom-GLOB-v02.0-fv02.0.nc --var analysed_sst

::

    <xarray.DataArray 'analysed_sst' (time: 1, lat: 720, lon: 1440)>
    [1036800 values with dtype=float32]
    Coordinates:
      * lat      (lat) float32 -89.875 -89.625 -89.375 ... 89.375 89.625 89.875
      * lon      (lon) float32 0.125 0.375 0.625 0.875 ... 359.375 359.625 359.875
      * time     (time) object 2017-06-05 12:00:00
    Attributes:
        long_name:      analysed sea surface temperature
        standard_name:  sea_surface_foundation_temperature
        type:           foundation
        units:          kelvin
        valid_min:      -300
        valid_max:      4500
        source:         UKMO-L4HRfnd-GLOB-OSTIA
        comment:


For creating a toy xcube dataset you can execute the command-line below. Please adjust the paths to your needs:

::

    $ xcube gen -o "your/output/path/demo_SST_xcube.zarr" -c examples/gen/config_files/xcube_sst_demo_config.yml --sort examples/gen/data/*.nc

The `configuration file`_ specifies the input processor, which in this case is the default one.
The output size is 10240, 5632. The bounding box of the data cube is given by ``output_region`` in the configuration file.
The output format (``output_writer_name``) is defined as well.
The chunking of the dimensions can be set by the ``chunksizes`` attribute of the ``output_writer_params`` parameter,
and in the example configuration file the chunking is set for latitude and longitude. If the chunking is not set, a automatic chunking is applied.
The spatial resampling method (``output_resampling``) is set to 'nearest' and the configuration file contains only one
variable which will be included into the xcube dataset - 'analysed-sst'.

The Analysed Sea Surface Temperature data set contains the variable already as needed. This means no pixel 
masking needs to be applied. However, this might differ depending on the input data. You can take a look at a 
`configuration file which takes Sentinel-3 Ocean and Land Colour Instrument (OLCI)`_
as input files, which is a bit more complex.
The advantage of using pixel expressions is, that the generated cube contains only valid pixels and the user of the
data cube does not have to worry about something like land-masking or invalid values.
Furthermore, the generated data cube is spatially regular. This means the data are aligned on a common spatial grid and
cover the same region. The time stamps are kept from the input data set.

**Caution:** If you have input data that has file names not only varying with the time stamp but with e.g. A and B as well,
you need to pass the input files in the desired order via a text file. Each line of the text file should contain the 
path to one input file. If you pass the input files in a desired order, then do not use the parameter ``--sort`` within
the commandline interface.


Optimizing and pruning a xcube dataset
======================================

If you want to optimize your generated xcube dataset e.g. for publishing it in a xcube viewer via xcube serve
you can use  :doc:`cli/xcube optimize`:

::

    $ xcube optimize demo_SST_xcube.zarr -C

By executing the command above, an optimized xcube dataset called demo_SST_xcube-optimized.zarr will be created.
You can take a look into the directory of the original xcube dataset and the optimized one, and you will notice that
a file called .zmetadata. .zmetadata contains the information stored in .zattrs and .zarray of each variable of the
xcube dataset and makes requests of metadata faster. The option ``-C`` optimizes coordinate variables by converting any
chunked arrays into single, non-chunked, contiguous arrays.

For deleting empty chunks :doc:`cli/xcube prune` can be used. It deletes all data files associated with empty (NaN-only)
chunks of an xcube dataset, and is restricted to the ZARR format.

::

    $ xcube prune demo_SST_xcube-optimized.zarr

The pruned xcube dataset is saved in place and does not need an output path. The size of the xcube dataset was 6,8 MB before pruning it
and 6,5 MB afterwards. According to the output printed to the terminal, 30 block files were deleted.

The metadata of the xcube dataset can be viewed with :doc:`cli/xcube dump` as well:

::

    $ xcube dump demo_SST_xcube-optimized.zarr

::

    <xarray.Dataset>
    Dimensions:       (bnds: 2, lat: 5632, lon: 10240, time: 3)
    Coordinates:
      * lat           (lat) float64 62.67 62.66 62.66 62.66 ... 48.01 48.0 48.0
        lat_bnds      (lat, bnds) float64 dask.array<shape=(5632, 2), chunksize=(5632, 2)>
      * lon           (lon) float64 -16.0 -16.0 -15.99 -15.99 ... 10.66 10.66 10.67
        lon_bnds      (lon, bnds) float64 dask.array<shape=(10240, 2), chunksize=(10240, 2)>
      * time          (time) datetime64[ns] 2017-06-05T12:00:00 ... 2017-06-07T12:00:00
        time_bnds     (time, bnds) datetime64[ns] dask.array<shape=(3, 2), chunksize=(3, 2)>
    Dimensions without coordinates: bnds
    Data variables:
        analysed_sst  (time, lat, lon) float64 dask.array<shape=(3, 5632, 10240), chunksize=(1, 704, 640)>
    Attributes:
        acknowledgment:             Data Cube produced based on data provided by ...
        comment:
        contributor_name:
        contributor_role:
        creator_email:              info@brockmann-consult.de
        creator_name:               Brockmann Consult GmbH
        creator_url:                https://www.brockmann-consult.de
        date_modified:              2019-09-25T08:50:32.169031
        geospatial_lat_max:         62.666666666666664
        geospatial_lat_min:         48.0
        geospatial_lat_resolution:  0.002604166666666666
        geospatial_lat_units:       degrees_north
        geospatial_lon_max:         10.666666666666664
        geospatial_lon_min:         -16.0
        geospatial_lon_resolution:  0.0026041666666666665
        geospatial_lon_units:       degrees_east
        history:                    xcube/reproj-snap-nc
        id:                         demo-bc-sst-sns-l2c-v1
        institution:                Brockmann Consult GmbH
        keywords:
        license:                    terms and conditions of the DCS4COP data dist...
        naming_authority:           bc
        processing_level:           L2C
        project:                    xcube
        publisher_email:            info@brockmann-consult.de
        publisher_name:             Brockmann Consult GmbH
        publisher_url:              https://www.brockmann-consult.de
        references:                 https://dcs4cop.eu/
        source:                     CMEMS Global SST & Sea Ice Anomaly Data Cube
        standard_name_vocabulary:
        summary:
        time_coverage_end:          2017-06-08T00:00:00.000000000
        time_coverage_start:        2017-06-05T00:00:00.000000000
        title:                      CMEMS Global SST Anomaly Data Cube

The metadata for the variable analysed_sst can be viewed:

::

    $ xcube dump demo_SST_xcube-optimized.zarr --var analysed_sst

::

    <xarray.DataArray 'analysed_sst' (time: 3, lat: 5632, lon: 10240)>
    dask.array<shape=(3, 5632, 10240), dtype=float64, chunksize=(1, 704, 640)>
    Coordinates:
      * lat      (lat) float64 62.67 62.66 62.66 62.66 ... 48.01 48.01 48.0 48.0
      * lon      (lon) float64 -16.0 -16.0 -15.99 -15.99 ... 10.66 10.66 10.66 10.67
      * time     (time) datetime64[ns] 2017-06-05T12:00:00 ... 2017-06-07T12:00:00
    Attributes:
        comment:
        long_name:           analysed sea surface temperature
        source:              UKMO-L4HRfnd-GLOB-OSTIA
        spatial_resampling:  Nearest
        standard_name:       sea_surface_foundation_temperature
        type:                foundation
        units:               kelvin
        valid_max:           4500
        valid_min:           -300
