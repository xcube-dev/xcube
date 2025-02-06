# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.


class DataStoreError(Exception):
    """Raised on error in any of the data store, opener, or writer methods.

    Args:
        message: The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
