# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Dict, Union

from ...util.config import to_name_dict_pairs, flatten_dict, load_configs


def get_config_dict(config_obj: Dict[str, Union[str, bool, int, float, list, dict, tuple]]):
    """
    Get configuration dictionary.

    :param config_obj: A configuration object.
    :return: Configuration dictionary
    :raise OSError, ValueError
    """
    config_file = config_obj.get("config_file")
    input_paths = config_obj.get("input_paths")
    input_processor = config_obj.get("input_processor")
    output_path = config_obj.get("output_path")
    output_writer = config_obj.get("output_writer")
    output_size = config_obj.get("output_size")
    output_region = config_obj.get("output_region")
    output_variables = config_obj.get("output_variables")
    output_resampling = config_obj.get("output_resampling")
    append_mode = config_obj.get("append_mode")
    sort_mode = config_obj.get("sort_mode")

    config = load_configs(*config_file) if config_file else {}

    # Overwrite current configuration by cli arguments
    if input_paths is not None and 'input_paths' not in config:
        if len(input_paths) == 1 and input_paths[0].endswith(".txt"):
            with open(input_paths[0]) as input_txt:
                input_paths = input_txt.readlines()
            config['input_paths'] = [x.strip() for x in input_paths]
        else:
            config['input_paths'] = input_paths

    if input_processor is not None:
        config['input_processor'] = input_processor

    if output_path is not None and 'output_path' not in config:
        config['output_path'] = output_path

    if output_writer is not None and 'output_writer' not in config:
        config['output_writer'] = output_writer

    if output_resampling is not None and 'output_resampling' not in config:
        config['output_resampling'] = output_resampling

    if output_size is not None:
        try:
            output_size = list(map(lambda c: int(c), output_size.split(',')))
        except ValueError:
            output_size = None
        if output_size is None or len(output_size) != 2:
            raise ValueError(f'output_size must have the form <width>,<height>,'
                             f' where both values must be positive integer numbers')
        config['output_size'] = output_size

    if output_region is not None:
        try:
            output_region = list(map(lambda c: float(c), output_region.split(',')))
        except ValueError:
            output_region = None
        if output_region is None or len(output_region) != 4:
            raise ValueError(f'output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,'
                             f' where all four numbers must be floating point numbers in degrees')
        config['output_region'] = output_region

    if output_variables is not None:
        output_variables = list(map(lambda c: c.strip(), output_variables.split(',')))
        if output_variables == [''] or any([var_name == '' for var_name in output_variables]):
            raise ValueError('output_variables must be a list of existing variable names')
        config['output_variables'] = output_variables

    if append_mode is not None and config.get('append_mode') is None:
        config['append_mode'] = append_mode

    if sort_mode is not None and config.get('sort_mode') is None:
        config['sort_mode'] = sort_mode

    processed_variables = config.get('processed_variables')
    if processed_variables:
        config['processed_variables'] = to_name_dict_pairs(processed_variables)

    output_variables = config.get('output_variables')
    if output_variables:
        config['output_variables'] = to_name_dict_pairs(output_variables, default_key='name')

    output_metadata = config.get('output_metadata')
    if output_metadata:
        config['output_metadata'] = flatten_dict(output_metadata)
    return config


