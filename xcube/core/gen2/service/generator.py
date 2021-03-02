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

import time
from typing import Type, TypeVar, Dict

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

    def __init__(self,
                 gen_config: CubeGeneratorRequest,
                 service_config: ServiceConfig,
                 progress_period: float = 1.0,
                 verbose: bool = False):
        assert_instance(gen_config, CubeGeneratorRequest, 'gen_config')
        assert_instance(service_config, ServiceConfig, 'service_config')
        assert_instance(progress_period, (int, float), 'progress_period')
        self._gen_config = gen_config
        self._service_config = service_config
        self._access_token = service_config.access_token
        self._progress_period = progress_period
        self._verbose = verbose

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
            request = {
                "audience": self._service_config.endpoint_url,
                "client_id": self._service_config.client_id,
                "client_secret": self._service_config.client_secret,
                "grant_type": "client-credentials",
            }
            # self.__dump_json(request)
            response = requests.post(self.endpoint_op('oauth/token'),
                                     json=request,
                                     headers=_BASE_HEADERS)
            token_response: Token = self._parse_response(response, Token)
            self._access_token = token_response.access_token
        return self._access_token

    def get_cube_info(self) -> CubeInfoWithCosts:
        response = requests.post(self.endpoint_op('cubegens/info'),
                                 json=self._gen_config.to_dict(),
                                 headers=self.auth_headers)
        return self._parse_response(response, CubeInfoWithCosts)

    def generate_cube(self):
        request = self._gen_config.to_dict()
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

    @classmethod
    def _get_cube_generation_result(cls, response: requests.Response) -> Result:
        response_instance: Response = cls._parse_response(response, Response)
        if response_instance.result.status.failed:
            raise CubeGeneratorError('Cube generation failed',
                                     remote_traceback=response_instance.traceback)
        return response_instance.result

    @classmethod
    def _parse_response(cls, response: requests.Response, response_type: Type[R]) -> R:
        CubeGeneratorError.maybe_raise_for_response(response)
        data = response.json()
        # cls.__dump_json(data)
        # noinspection PyBroadException
        try:
            return response_type.from_dict(data)
        except Exception as e:
            raise RuntimeError(f'internal error: unexpected response'
                               f' from API call {response.url}: {e}') from e

    @classmethod
    def __dump_json(cls, obj):
        import json
        import sys
        json.dump(obj, sys.stdout, indent=2)
