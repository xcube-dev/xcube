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

import json
import time
from typing import Type, TypeVar, Dict, Any

import requests

from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .config import ServiceConfig
from .response import CubeInfoWithCosts
from .response import Response
from .response import Result
from .response import Token
from ..error import CubeGeneratorError
from ..generator import CubeGenerator
from ..request import CubeGeneratorRequest

_BASE_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}

R = TypeVar('R')


class CubeGeneratorService(CubeGenerator):
    """
    Service for generating data cubes.

    Creates cube views from one or more cube stores, resamples them to a
    common grid, optionally performs some cube transformation, and writes
    the resulting cube to some target cube store.

    :param request: Cube generation request.
    :param service_config: An service configuration object.
    :param verbosity: Level of verbosity, 0 means off.
    """

    def __init__(self,
                 request: CubeGeneratorRequest,
                 service_config: ServiceConfig,
                 progress_period: float = 1.0,
                 verbosity: int = 0):
        assert_instance(request, CubeGeneratorRequest, 'request')
        assert_instance(service_config, ServiceConfig, 'service_config')
        assert_instance(progress_period, (int, float), 'progress_period')
        self._request = request
        self._service_config = service_config
        self._access_token = service_config.access_token
        self._progress_period = progress_period
        self._verbosity = verbosity

    def endpoint_op(self, op_path: str) -> str:
        return f'{self._service_config.endpoint_url}{op_path}'

    @property
    def auth_headers(self) -> Dict:
        return {
            **_BASE_HEADERS,
            'Authorization': f'Bearer {self.access_token}',
        }

    @property
    def access_token(self) -> str:
        if self._access_token is None:
            request_data = {
                "audience": self._service_config.endpoint_url,
                "client_id": self._service_config.client_id,
                "client_secret": self._service_config.client_secret,
                "grant_type": "client-credentials",
            }
            # self.__dump_json(request)
            response = requests.post(self.endpoint_op('oauth/token'),
                                     json=request_data,
                                     headers=_BASE_HEADERS)
            token_response: Token = self._parse_response(response, Token, request_data=request_data)
            self._access_token = token_response.access_token
        return self._access_token

    def get_cube_info(self) -> CubeInfoWithCosts:
        request_data = self._request.to_dict()
        response = requests.post(self.endpoint_op('cubegens/info'),
                                 json=request_data,
                                 headers=self.auth_headers)
        return self._parse_response(response, CubeInfoWithCosts, request_data=request_data)

    def generate_cube(self):
        request = self._request.to_dict()
        # self.__dump_json(request)
        response = requests.put(self.endpoint_op('cubegens'),
                                json=request,
                                headers=self.auth_headers)
        result = self._get_cube_generation_result(response)
        cubegen_id = result.cubegen_id

        last_worked = 0
        with observe_progress('Generating cube', 100) as cm:
            while True:
                time.sleep(self._progress_period)

                response = requests.get(self.endpoint_op(f'cubegens/{cubegen_id}'),
                                        headers=self.auth_headers)
                result = self._get_cube_generation_result(response)
                if result.status.succeeded:
                    return

                if result.progress is not None and len(result.progress) > 0:
                    progress_state = result.progress[0].state
                    total_work = progress_state.total_work
                    progress = progress_state.progress or 0
                    worked = progress * total_work
                    work = 100 * ((worked - last_worked) / total_work)
                    if work > 0:
                        cm.worked(work)
                        last_worked = worked

    def _get_cube_generation_result(self, response: requests.Response,
                                    request_data: Dict[str, Any] = None) -> Result:
        response_instance: Response = self._parse_response(response,
                                                           Response,
                                                           request_data=request_data)
        result = response_instance.result
        if result.status.failed:
            message = 'Cube generation failed'
            if result.status.conditions:
                sub_messages = [item['message'] or '' for item in result.status.conditions
                                if isinstance(item, dict) and 'message' in item]
                message = f'{message}: {": ".join(sub_messages)}'
            raise CubeGeneratorError(message,
                                     remote_output=result.output,
                                     remote_traceback=response_instance.traceback)
        return result

    def _parse_response(self,
                        response: requests.Response,
                        response_type: Type[R],
                        request_data: Dict[str, Any] = None) -> R:
        CubeGeneratorError.maybe_raise_for_response(response)
        response_data = response.json()
        if self._verbosity >= 3:
            self.__dump_json(response.request.method, response.url, request_data, response_data)

        # noinspection PyBroadException
        try:
            return response_type.from_dict(response_data)
        except Exception as e:
            raise RuntimeError(f'internal error: unexpected response'
                               f' from API call {response.url}: {e}') from e

    @classmethod
    def __dump_json(cls, method, url, request_data, response_data):
        """
        Dump debug info as JSON to stdout.

        Used for debugging only.
        """
        url_line = f'{method} {url}:'
        request_line = 'Request:'
        response_line = 'Response:'

        print('=' * len(url_line))
        print(url_line)
        print('=' * len(url_line))
        print('-' * len(request_line))
        print(request_line)
        print('-' * len(request_line))
        print(json.dumps(request_data, indent=2))
        print('-' * len(response_line))
        print(response_line)
        print('-' * len(response_line))
        print(json.dumps(response_data, indent=2))
