# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import click

__author__ = "Norman Fomferra (Brockmann Consult GmbH)"


@click.command(name="benchmark", hidden=True)
@click.argument("config")
@click.option(
    "--repeats",
    "-R",
    metavar="REPEATS",
    type=int,
    default=1,
    help="Number of repetitions",
)
def benchmark(config: str, repeats: int):
    """
    Measure runtime performance of an executable.
    Record the processing time of the executable with varying parameters defined CONFIG.
    Repeat the executions REPEATS times.

    CONFIG is a YAML file that specifies the command to be executed and the parameter combinations
    to be benchmarked. The CONFIG

    \b
    command: my_exec --p1 ${param_1} --p2 ${param_2}
    params:
    - param_1
    - param_2
    param_1: [1, 2, 3]
    param_2: ['a', 'b']

    will benchmark all combinations of my_exec:

    \b
    my_exec --p1 1 --p2 a
    my_exec --p1 1 --p2 b
    my_exec --p1 2 --p2 a
    my_exec --p1 2 --p2 b
    my_exec --p1 3 --p2 a
    my_exec --p1 3 --p2 b
    """

    config_path = config
    repetition_count = repeats

    import yaml

    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    import itertools

    command_template = config["command"]
    param_names = config["params"]

    param_values = [config[param_name] for param_name in param_names]

    import numpy as np

    param_values_combinations = list(itertools.product(*param_values))
    param_combination_count = len(param_values_combinations)
    times = np.ndarray((repetition_count, param_combination_count), dtype=np.float64)

    import subprocess
    import time
    import sys

    for repetition_index in range(repetition_count):
        for param_combination_index in range(param_combination_count):
            param_values = param_values_combinations[param_combination_index]
            params = list(zip(param_names, param_values))
            if isinstance(command_template, str):
                command = _apply_template(command_template, params)
            else:
                command = [
                    _apply_template(arg_template, params)
                    for arg_template in command_template
                ]

            try:
                t0 = time.perf_counter()
                subprocess.check_call(command)
                t1 = time.perf_counter()
                times[repetition_index, param_combination_index] = t1 - t0
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                times[repetition_index, param_combination_index] = np.nan

    print(f"# command template: {command_template}")
    print(f"# repetition count: {repetition_count}")
    if repetition_count > 1:
        times_mean = np.nanmean(times, axis=0)
        times_median = np.nanmedian(times, axis=0)
        times_stdev = np.nanstd(times, axis=0)
        times_min = np.nanmin(times, axis=0)
        times_max = np.nanmax(times, axis=0)
        print(
            f"id;"
            f"{';'.join(param_names)};"
            f"time-mean;"
            f"time-median;"
            f"time-stdev;"
            f"time-min;"
            f"time-max"
        )
        for param_combination_index in range(param_combination_count):
            param_values = param_values_combinations[param_combination_index]
            print(
                f"{param_combination_index};"
                f"{';'.join(param_values)};"
                f"{times_mean[param_combination_index]};"
                f"{times_median[param_combination_index]};"
                f"{times_stdev[param_combination_index]};"
                f"{times_min[param_combination_index]};"
                f"{times_max[param_combination_index]}"
            )
    else:
        print(f"id;" f"{';'.join(param_names)};" f"time")
        for param_combination_index in range(param_combination_count):
            param_values = param_values_combinations[param_combination_index]
            print(
                f"{param_combination_index};"
                f"{';'.join(param_values)};"
                f"{times[0, param_combination_index]}"
            )


def _apply_template(template, params) -> str:
    result = template
    for param_name, param_value in params:
        result = result.replace("${" + param_name + "}", str(param_value))
    return result
