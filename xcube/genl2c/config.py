# The MIT License (MIT)
# Copyright (c) 2018 by the xcube development team and contributors
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

import yaml

from xcube.config import to_name_dict_pairs, flatten_dict


def get_config_dict(config_obj, open_function):
    """
    Get configuration dictionary.

    :param config_obj: A configuration object.
    :param open_function: Function used to open YAML configuration file.
    :return: Configuration dictionary
    :raise OSError, ValueError
    """
    config_file = config_obj.config_file
    input_files = config_obj.input_files
    input_type = config_obj.input_type
    output_dir = config_obj.output_dir
    output_name = config_obj.output_name
    output_format = config_obj.output_format
    output_size = config_obj.output_size
    output_region = config_obj.output_region
    output_variables = config_obj.output_variables
    output_resampling = config_obj.output_resampling

    if config_file is not None:
        try:
            with open_function(config_file) as stream:

                config = yaml.load(stream)
        except yaml.YAMLError as e:
            raise ValueError(f'YAML in {config_file!r} is invalid: {e}') from e
        except OSError as e:
            raise ValueError(f'cannot load configuration from {config_file!r}: {e}') from e
    else:
        config = {}

    if input_files is not None:
        config['input_files'] = input_files

    if input_type is not None:
        config['input_type'] = input_type

    if output_dir is not None:
        config['output_dir'] = output_dir

    if output_name is not None:
        config['output_name'] = output_name

    if output_format is not None:
        config['output_format'] = output_format

    if output_resampling is not None:
        config['output_resampling'] = output_resampling

    if output_size is not None:
        try:
            output_size = list(map(lambda c: int(c), output_size.split(',')))
        except ValueError:
            output_size = None
        if output_size is None or len(output_size) != 2:
            raise ValueError(f'invalid output_size {config_obj.output_size!r}')
        config['output_size'] = output_size

    if output_region is not None:
        try:
            output_region = list(map(lambda c: float(c), output_region.split(',')))
        except ValueError:
            output_region = None
        if output_region is None or len(output_region) != 4:
            raise ValueError(f'invalid output_region {config_obj.output_region!r}')
        config['output_region'] = output_region

    if output_variables is not None:
        output_variables = list(map(lambda c: c.strip(), output_variables.split(',')))
        if output_variables == ['']:
            raise ValueError('output_variables must contain at least one name')
        if any([var_name == '' for var_name in output_variables]):
            raise ValueError('all names in output_variables must be non-empty')
        config['output_variables'] = output_variables

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
