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
    Example for a configuration file: [dcs4cop-config.yml](config_files/dcs4cop-config.yml)
      
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

For creating a little cube you can execute the commandline below with the paths adjusted to your needs.


`xcube gen  examples/data/*.nc --dir "your/output/path/" -c examples/config_files/dcs4cop-gen_BC_config_CMEMS.yml -a --sort`

Caution: If you have input data that has file names not only varying with the time stamp but with e.g. A and B as well, 
you need to pass the input files in the desired order via a text file. Each line of the text file should contain the 
path to one input file. If you pass the input files in a desired order, then do not use the parameter `--sort` within
the commandline interface