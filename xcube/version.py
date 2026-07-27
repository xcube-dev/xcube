# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from importlib.metadata import PackageNotFoundError, version as get_version

try:
    # xcube on conda-forge and editable pip installs
    version = get_version("xcube")
except PackageNotFoundError:
    # On PyPI, xcube is called xcube-core
    version = get_version("xcube-core")
