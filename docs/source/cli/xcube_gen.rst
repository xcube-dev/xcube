.. _`CF Conventions`: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html
.. _`numpy ufuncs`: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`xarray.DataArray API`: http://xarray.pydata.org/en/stable/api.html#dataarray
.. _`YAML format`: https://en.wikipedia.org/wiki/YAML

=============
``xcube gen``
=============

Synopsis
========

Generate xcube dataset.

::

    $ xcube gen --help

::

    Usage: xcube gen [OPTIONS] [INPUT]...

      Generate xcube dataset. Data cubes may be created in one go or successively
      for all given inputs. Each input is expected to provide a single time slice
      which may be appended, inserted or which may replace an existing time slice
      in the output dataset. The input paths may be one or more input files or a
      pattern that may contain wildcards '?', '*', and '**'. The input paths can
      also be passed as lines of a text file. To do so, provide exactly one input
      file with ".txt" extension which contains the actual input paths to be used.

    Options:
      -P, --proc INPUT-PROCESSOR      Input processor name. The available input
                                      processor names and additional information
                                      about input processors can be accessed by
                                      calling xcube gen --info . Defaults to
                                      "default", an input processor that can deal
                                      with simple datasets whose variables have
                                      dimensions ("lat", "lon") and conform with
                                      the CF conventions.
      -c, --config CONFIG             xcube dataset configuration file in YAML
                                      format. More than one config input file is
                                      allowed.When passing several config files,
                                      they are merged considering the order passed
                                      via command line.
      -o, --output OUTPUT             Output path. Defaults to 'out.zarr'
      -f, --format FORMAT             Output format. Information about output
                                      formats can be accessed by calling xcube gen
                                      --info. If omitted, the format will be
                                      guessed from the given output path.
      -S, --size SIZE                 Output size in pixels using format
                                      "<width>,<height>".
      -R, --region REGION             Output region using format "<lon-min>,<lat-
                                      min>,<lon-max>,<lat-max>"
      --variables, --vars VARIABLES   Variables to be included in output. Comma-
                                      separated list of names which may contain
                                      wildcard characters "*" and "?".
      --resampling [Average|Bilinear|Cubic|CubicSpline|Lanczos|Max|Median|Min|Mode|Nearest|Q1|Q3]
                                      Fallback spatial resampling algorithm to be
                                      used for all variables. Defaults to
                                      'Nearest'. The choices for the resampling
                                      algorithm are: ['Average', 'Bilinear',
                                      'Cubic', 'CubicSpline', 'Lanczos', 'Max',
                                      'Median', 'Min', 'Mode', 'Nearest', 'Q1',
                                      'Q3']
      -a, --append                    Deprecated. The command will now always
                                      create, insert, replace, or append input
                                      slices.
      --prof                          Collect profiling information and dump
                                      results after processing.
      --no_sort                       The input file list will not be sorted
                                      before creating the xcube dataset. If
                                      --no_sort parameter is passed, the order of
                                      the input list will be kept. This parameter
                                      should be used for better performance,
                                      provided that the input file list is in
                                      correct order (continuous time).
      -I, --info                      Displays additional information about format
                                      options or about input processors.
      --dry_run                       Just read and process inputs, but don't
                                      produce any outputs.
      --help                          Show this message and exit.



Below is the ouput of a ``xcube gen --info`` call showing five input processors installed via plugins.

::

    $ xcube gen --info

::

    input processors to be used with option --proc:
      default                           Single-scene NetCDF/CF inputs in xcube standard format
      rbins-seviri-highroc-scene-l2     RBINS SEVIRI HIGHROC single-scene Level-2 NetCDF inputs
      rbins-seviri-highroc-daily-l2     RBINS SEVIRI HIGHROC daily Level-2 NetCDF inputs
      snap-olci-highroc-l2              SNAP Sentinel-3 OLCI HIGHROC Level-2 NetCDF inputs
      snap-olci-cyanoalert-l2           SNAP Sentinel-3 OLCI CyanoAlert Level-2 NetCDF inputs
      vito-s2plus-l2                    VITO Sentinel-2 Plus Level 2 NetCDF inputs

    For more input processors use existing "xcube-gen-..." plugins from the github organisation DCS4COP or write own plugin.


    Output formats to be used with option --format:
      zarr                    (*.zarr)      Zarr file format (http://zarr.readthedocs.io)
      netcdf4                 (*.nc)        NetCDF-4 file format
      csv                     (*.csv)       CSV file format
      mem                     (*.mem)       In-memory dataset I/O


Configuration File
==================

Configuration files passed to ``xcube gen`` via the ``-c, --config`` option use `YAML format`_.
Multiple configuration files may be given. In this case all configurations are merged into a single one.
Parameter values will be overwritten by subsequent configurations if they are scalars. If
they are objects / mappings, their values will be deeply merged.

The following parameters can be used in the configuration files:

``input_processor`` : str
    The name of an *input processor*. See ``-P, --proc`` option above.

    :Default: The default value is ``'default'``, xcube's default input processor. It can ingest and process
        inputs that

        * use an ``EPSG:4326`` (or compatible) grid;
        * have 1-D ``lon`` and ``lat`` coordinate variables using WGS84 coordinates and decimal degrees;
        * have a decodable 1-D ``time`` coordinate or define the one of the following global attribute pairs
          ``time_coverage_start`` and ``time_coverage_end``,
          ``time_start`` and ``time_end`` or ``time_stop``;
        * provide data variables with the dimensions ``time``, ``lat``, ``lon``, in this order.
        * conform to the `CF Conventions`_.

``output_size`` : [int, int]
    The spatial dimension sizes of the output dataset given as number of grid
    cells in longitude and latitude direction (width and height).

``output_region`` : [float, float, float, float]
    The spatial extent of output datasets given as a bounding box [lat-min, lat-min, lon-max, lat-max]
    using decimal degrees.

``output_variables`` : [*variable-definitions*]
    The definition of variables that will be included in the output dataset.
    Each variable definition may be just a name or a mapping from a name to variable attributes.
    If it is just a name it must be the name of an existing variable either in the INPUT
    or in ``processed_variables``. If the variable definition is a mapping, some of the
    attributes affect the way how variables are processed.
    All but the ``name`` attributes become variable metadata in the output.

    ``name`` : str
        The new name of the variable in the output.

    ``valid_pixel_expression`` : str
        An expression used to mask this variable, see :ref:`expressions`. The expression identifies all
        valid pixels in each INPUT.

    ``resampling`` : str
        The resampling method used. See ``--resampling`` option above.

    :Default: By default, all variables in INPUT will occur in output.


``processed_variables`` : [*variable-definitions*]
    The definition of variables that will be produced or processed
    after reading each INPUT. The main purpose is to generate intermediate variables that can be referred to in
    the ``expression`` in other variable definitions in ``processed_variables`` and
    ``valid_pixel_expression`` in variable definitions in ``output_variables``. The following attributes are
    recognised:

    ``expression`` : str
        An expression used to produce this variable, see :ref:`expressions`.

``output_writer_name`` : str
    The name of a supported output format. May be one of ``'zarr'``, ``'netcdf4'``, ``'mem'``.

    :Default: ``'zarr'``

``output_writer_params`` : str
   A mapping that defines parameters that are passed to output writer denoted by ``output_writer_name``.
   Through the ``output_writer_params`` a packing of the variables may be defined.
   If not specified the default does not apply any packing which results in: ::

        _FillValue:  nan
        dtype:       dtype('float32')

   and for coordinate variables ::


        dtype:       dtype('int64')

   The user may specify a different packing variables,
   which might be useful for reducing the storage size of the datacubes.
   Currently it is only implemented for zarr format.
   This may be done by passing the parameters for packing as the following: ::

      output_writer_params:

        packing:
          analysed_sst:
            scale_factor: 0.07324442274239326
            add_offset: -300.0
            dtype: 'uint16'
            _FillValue: 0.65535


   Furthermore the compressor may be defined as well by,
   if not specified the default compressor
   (cname='lz4', clevel=5, shuffle=SHUFFLE, blocksize=0) is used. ::


      output_writer_params:

        compressor:
          cname: 'zstd'
          clevel: 1
          shuffle: 2


``output_metadata`` : [*attribute-definitions*]
    General metadata that will be present in the output dataset as global attributes.
    You can put any common
    `CF attributes <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#attribute-appendix>`_
    here.

    Any attributes that are mappings will be "flattened" by concatenating the attribute names using
    the underscrore character. For example,::

      publisher:
        name:  "Brockmann Consult GmbH"
        url:   "https://www.brockmann-consult.de"

    will create the two entries::

      publisher_name:  "Brockmann Consult GmbH"
      publisher_url:   "https://www.brockmann-consult.de"


.. _expressions:

Expressions
===========

Expressions are plain text values of the ``expression`` and ``valid_pixel_expression`` attributes of the
variable definitions in the ``processed_variables`` and ``output_variables`` parameters.
The expression syntax is that of standard Python.
``xcube gen`` uses expressions to produce new variables listed in ``processed_variables`` and to mask
variables by the ``valid_pixel_expression``.


An expression may refer any variables in the INPUT datasets and any variables defined by the ``processed_variables``
parameter. Expressions may make use of most of the standard Python operators
and may apply all `numpy ufuncs`_  to referred variables. Also most of the `xarray.DataArray API`_
may be used on variables within an expression.

In order to utilise flagged variables, the syntax ``variable_name.flag_name`` can be used in expressions.
According to the `CF Conventions <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#flags>`_,
flagged variables are variables whose metadata include the attributes ``flag_meanings`` and ``flag_values``
and/or ``flag_masks``. The ``flag_meanings`` attribute enumerates the allowed values for ``flag_name``.
The flag attributes must be present in the variables of each INPUT.


Example
=======

An example that uses a configuration file only::

    $ xcube gen --config ./config.yml /data/eo-data/SST/2018/**/*.nc

An example that uses the default input processor and passes all other configuration via command-line options::

    $ xcube gen -S 2000,1000 -R 0,50,5,52.5 --vars conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags -o hiroc-cube.zarr /data/eo-data/SST/2018/**/*.nc


Some input processors have been developed for specific EO data sources 
used within the DCS4COP project. They may serve as examples how to develop
input processor plug-ins:

* `xcube-gen-rbins <https://github.com/dcs4cop/xcube-gen-rbins>`_
* `xcube-gen-bc <https://github.com/dcs4cop/xcube-gen-bc>`_
* `xcube-gen-vito <https://github.com/dcs4cop/xcube-gen-vito>`_

Python API
==========

The related Python API function is :py:func:`xcube.core.gen.gen.gen_cube`.


