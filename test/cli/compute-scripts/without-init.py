# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


# noinspection PyUnusedLocal
def compute(variable_a, variable_b, input_params=None, **kwargs):
    a = input_params.get("a", 0.5)
    b = input_params.get("b", 0.5)
    return a * variable_a + b * variable_b
