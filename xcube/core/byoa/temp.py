# The MIT License (MIT)
# Copyright (c) 2021 by the xcube development team and contributors
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

import atexit
import os
import shutil
import tempfile

from .constants import DEFAULT_TEMP_FILE_PREFIX


def new_temp_dir(suffix: str = '') -> str:
    """
    Create new temporary directory
    that will be removed when the process ends.
    """
    return remove_dir_later(
        tempfile.mkdtemp(prefix=DEFAULT_TEMP_FILE_PREFIX,
                         suffix=suffix)
    )


def new_temp_file(suffix: str = '') -> str:
    """
    Create new temporary file
    that will be removed when the process ends.
    """
    return remove_file_later(
        tempfile.mktemp(prefix=DEFAULT_TEMP_FILE_PREFIX,
                        suffix=suffix)
    )


def remove_dir_later(dir_path: str) -> str:
    """Remove directory when the process ends."""
    def _remove_dir_later():
        shutil.rmtree(dir_path, ignore_errors=True)

    atexit.register(_remove_dir_later)
    return dir_path


def remove_file_later(file_path: str) -> str:
    """Remove file when the process ends."""
    def _remove_file_later():
        os.remove(file_path)

    atexit.register(_remove_file_later)
    return file_path
