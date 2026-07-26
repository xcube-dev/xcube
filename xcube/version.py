# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from importlib.metadata import version as get_version, PackageNotFoundError

try:
    version = get_version("xcube")
except PackageNotFoundError:
    version = get_version("xcube-core")
