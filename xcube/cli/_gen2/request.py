# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
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
import sys
import json
import os.path
from typing import Optional, Type, Dict

import yaml


class Request:
    pass

    @classmethod
    def from_dict(cls, request_dict: Dict, exception_type: Type[BaseException] = ValueError) -> 'Request':
        if not isinstance(request_dict, dict):
            raise exception_type(f'Invalid cube generation request.')
        if not request_dict:
            raise exception_type(f'Empty cube generation request.')
        # TODO: implement me
        raise NotImplementedError()

    @classmethod
    def from_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> 'Request':
        request_dict = cls._load_request_file(request_file, exception_type=exception_type)
        return cls.from_dict(request_dict)

    @classmethod
    def _load_request_file(cls, request_file: Optional[str], exception_type: Type[BaseException] = ValueError) -> Dict:

        if request_file is not None and not os.path.exists(request_file):
            raise exception_type(f'Cube generation request "{request_file}" not found.')

        try:
            if request_file is None:
                if not sys.stdin.isatty():
                    return json.load(sys.stdin)
            else:
                with open(request_file, 'r') as fp:
                    if request_file.endswith('.json'):
                        return json.load(fp)
                    else:
                        return yaml.safe_load(fp)
        except BaseException as e:
            raise exception_type(f'Error loading cube generation request "{request_file}": {e}') from e

        raise exception_type(f'Missing cube generation request.')