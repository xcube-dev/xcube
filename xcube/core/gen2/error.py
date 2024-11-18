# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Optional, List

from xcube.util.assertions import assert_instance


class CubeGeneratorError(ValueError):
    """Represent a client or server error that
    may occur in the data cube generators.

    Args:
        args: arguments passed to base exceptions.
        status_code: Optional status code of the error (an integer). If
            given, HTTP error codes should be used. The range 400-499
            indicates client errors. The range 500-599 indicates server
            errors.
        remote_traceback: Traceback of an error occurred in a remote
            process.
        remote_output: Terminal output of a remote process.
    """

    def __init__(
        self,
        *args,
        status_code: Optional[int] = None,
        remote_traceback: Optional[list[str]] = None,
        remote_output: Optional[list[str]] = None,
        **kwargs
    ):
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        if status_code is not None:
            assert_instance(status_code, int, "status_code")
        self._status_code = status_code
        self._remote_traceback = remote_traceback
        self._remote_output = remote_output

    @property
    def status_code(self) -> Optional[int]:
        """Status code of the error.
        May be None.
        """
        return self._status_code

    @property
    def remote_traceback(self) -> Optional[list[str]]:
        """Traceback of an error occurred in a remote process.
        May be None.
        """
        return self._remote_traceback

    @property
    def remote_output(self) -> Optional[list[str]]:
        """Terminal output of a remote process.
        May be None.
        """
        return self._remote_output
