# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.
from collections.abc import Sequence

from xcube.util.config import flatten_dict, load_configs, to_name_dict_pairs


def get_config_dict(
    config_files: Sequence[str] = None,
    input_paths: Sequence[str] = None,
    input_processor_name: str = None,
    output_path: str = None,
    output_writer_name: str = None,
    output_size: str = None,
    output_region: str = None,
    output_variables: str = None,
    output_resampling: str = None,
    append_mode: bool = True,
    profile_mode: bool = False,
    no_sort_mode: bool = False,
):
    """Get a configuration dictionary from given (command-line) arguments.

    Returns:
        Configuration dictionary

    Raises: OSError, ValueError
    """

    config = load_configs(*config_files) if config_files else {}

    # preserve backward compatibility for old names
    if "input_processor" in config:
        config["input_processor_name"] = config.pop("input_processor")
    if "output_writer" in config:
        config["output_writer_name"] = config.pop("output_writer")

    # Overwrite current configuration by cli arguments
    if input_paths is not None and "input_paths" not in config:
        if len(input_paths) == 1 and input_paths[0].endswith(".txt"):
            with open(input_paths[0]) as input_txt:
                input_paths = input_txt.readlines()
            config["input_paths"] = [x.strip() for x in input_paths]
        else:
            config["input_paths"] = input_paths

    if input_processor_name is not None and "input_processor_name" not in config:
        config["input_processor_name"] = input_processor_name

    if output_path is not None and "output_path" not in config:
        config["output_path"] = output_path

    if output_writer_name is not None and "output_writer_name" not in config:
        config["output_writer_name"] = output_writer_name

    if output_resampling is not None and "output_resampling" not in config:
        config["output_resampling"] = output_resampling

    if output_size is not None:
        try:
            output_size = list(map(lambda c: int(c), output_size.split(",")))
        except ValueError:
            output_size = None
        if output_size is None or len(output_size) != 2:
            raise ValueError(
                f"output_size must have the form <width>,<height>,"
                f" where both values must be positive integer numbers"
            )
        config["output_size"] = output_size

    if output_region is not None:
        try:
            output_region = list(map(lambda c: float(c), output_region.split(",")))
        except ValueError:
            output_region = None
        if output_region is None or len(output_region) != 4:
            raise ValueError(
                f"output_region must have the form <lon_min>,<lat_min>,<lon_max>,<lat_max>,"
                f" where all four numbers must be floating point numbers in degrees"
            )
        config["output_region"] = output_region

    if output_variables is not None:
        output_variables = list(map(lambda c: c.strip(), output_variables.split(",")))
        if output_variables == [""] or any(
            [var_name == "" for var_name in output_variables]
        ):
            raise ValueError(
                "output_variables must be a list of existing variable names"
            )
        config["output_variables"] = output_variables

    if profile_mode is not None and config.get("profile_mode") is None:
        config["profile_mode"] = profile_mode

    if append_mode is not None and config.get("append_mode") is None:
        config["append_mode"] = append_mode

    if no_sort_mode is not None and config.get("no_sort_mode") is None:
        config["no_sort_mode"] = no_sort_mode

    processed_variables = config.get("processed_variables")
    if processed_variables:
        config["processed_variables"] = to_name_dict_pairs(processed_variables)

    output_variables = config.get("output_variables")
    if output_variables:
        config["output_variables"] = to_name_dict_pairs(
            output_variables, default_key="name"
        )

    output_metadata = config.get("output_metadata")
    if output_metadata:
        config["output_metadata"] = flatten_dict(output_metadata)

    return config
