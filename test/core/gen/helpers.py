# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os


def get_inputdata_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "inputdata", name)
