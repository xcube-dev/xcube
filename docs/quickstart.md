# Steps to create own Data Cube

In order to create a suitable Data Cube for your needs, there are some questions which need to be answered before hand. 

1. Find appropriate grid for cube:
    * Which region needs to be covered by the data cube? 
    * At which resolution should the data cube be? 

    &rarr; Use `xcube grid` to determine a suitable bounding box which includes the region of interest 
    and is fixed to a global grid. 

2. Decide on variables to be included:
    * Are all variables needed? If not, select the ones to be included.
    * Should specific pixels be masked out based on pixel expressions? 

3. Decide on chunking:
    * The chunking should depend on your needs

    &rarr; This website might be helpful when deciding about the chunk sizes:  https://docs.dask.org/en/latest/array-chunks.html

4. Decide on the Data Cube output name and output path.

5. For creating a growing Data Cube with each input file, select the append mode. 
    
6. Use configuration file for generating your Data Cube:
    * You might not want to place all settings for your data cube within the command line, 
    you could use the parameter `-c, --config` and pass the above settings within a yaml-file. 
    Example for a configuration file: [dcs4cop-gen_BC_config_CMEMS.yml](../examples/config_files/dcs4cop-gen_BC_config_CMEMS.yml)
      
    * The parameter, which can be used within the configuration file are: 
        * input_files
        * input_processor 
        * output_dir 
        * output_name 
        * output_writer
        * output_size 
        * output_region 
        * output_variables
        * output_resampling 
        * append_mode 
        * sort_mode 
        
## Minimal working example

The input data for the examlpe contain sea surface temperature anomaly over the global ocean and are provided by [Copernicus
Marine Environment Monitoring Service](http://marine.copernicus.eu/). 
Here is a [Product User Manual](http://resources.marine.copernicus.eu/documents/PUM/CMEMS-SST-PUM-010-001.pdf) for the input data.

For creating a little cube you can execute the commandline below with the paths adjusted to your needs.

`xcube gen  examples/data/*.nc --dir "your/output/path/" -c examples/config_files/dcs4cop-gen_BC_config_CMEMS.yml -a --sort`

The [configuration file](../examples/config_files/dcs4cop-gen_BC_config_CMEMS.yml) specifies the input processor, 
which in this case is the default one. The output size is 10240, 5632 which was derived by using `xcube grid` 
for a spatial resolution of 300m and a bounding box (-15,48,10,62)(lon_min,lat_min,lon_max,lat_max). This also results 
in a adjusted bounding box which places the region into a global grid, called *output_region* in the configuration file. 
The output name (output_name) of the data cube and the format (output_writer) are defined as well. 
The chunking of the dimensions can be set by the output writer parameter (output_writer_params) called chunksizes, 
and here the chunking is set for latitude and longitude. If the chunking is not set, a automatic chunking is appllied.
The spatial resampling method (output_resampling) is set to 'nearest' and the confguration file contains only one 
variable which will be included into the data cube - 'analysed-sst'.

The Analysed Sea Surface Temperature data set contains the variable already as needed. This means no pixel 
masking needs to be applied. However, this might differ depending on the input data. You can take a look at a 
[configuration file which takes SENTINEL-3 Ocean and Land Colour Instrument (OLCI)](../examples/config_files/dcs4cop-config.yml)
as input files, which is a bit more complex.
The advantage of using pixel expressions is, that the generated cube contains only valid pixels and the user of the data cube
does not have to worry about something like land-masking or invalid values. 
Furthermore, the generated data cube is spatially regular meaning that for each time stamp the daca cells are located 
always at the same position. The time stamps are kept from the input data set. A possibility to generate data cubes with 
regular time dimension is under development. At the moment, time resampling can be conducted through importing the data cube 
with xarray into a Notebook (example will follow soon).

__Caution:__ If you have input data that has file names not only varying with the time stamp but with e.g. A and B as well, 
you need to pass the input files in the desired order via a text file. Each line of the text file should contain the 
path to one input file. If you pass the input files in a desired order, then do not use the parameter `--sort` within
the commandline interface.

A generated data cube can be viewed via the [xcube-viewer](https://github.com/dcs4cop/xcube-viewer/)
 and with the help of using `xcube serve`. 