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

import requests
import xarray as xr

from xcube.util.assertions import assert_instance
from xcube.util.progress import observe_progress
from .config import ServiceConfig
from ..config import CubeGeneratorConfig
from ..error import CubeGeneratorError
from ..generator import CubeGenerator

DEFAULT_ENDPOINT_URL = 'https://xcube-gen.brockmann-consult.de/api/v2'
# DEFAULT_ENDPOINT_URL = 'https://stage.xcube-gen.brockmann-consult.de/api/v2'

BASE_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class CubeGeneratorService(CubeGenerator):

    def __init__(self,
                 gen_config: CubeGeneratorConfig,
                 service_config: ServiceConfig,
                 verbose: bool = False):
        assert_instance(gen_config, CubeGeneratorConfig, 'gen_config')
        assert_instance(service_config, ServiceConfig, 'service_config')
        self._gen_config = gen_config
        self._service_config = service_config
        self._access_token = service_config.access_token
        self._verbose = verbose

    def endpoint_op(self, op_path: str) -> str:
        return f'{self._service_config.endpoint_url or DEFAULT_ENDPOINT_URL}/{op_path}'

    @property
    def access_token(self) -> str:
        if self._access_token is None:
            response = requests.post(self.endpoint_op('oauth/token'),
                                     json={
                                         "client_id": self._service_config.client_id,
                                         "client_secret": self._service_config.client_secret,
                                         # TODO: what to pass here?
                                         "audience": "audience",
                                         # TODO: what to pass here?
                                         "grant_type": "grant_type",
                                     },
                                     headers=BASE_HEADERS)
            self._maybe_raise(response)
            self._access_token = response.json().get('access_token')
        return self._access_token

    def generate_cube(self):
        response = requests.put(self.endpoint_op('cubegens'),
                                json=self._gen_config,
                                headers={
                                    **BASE_HEADERS,
                                    'Authorization': f'Bearer {self.access_token}',
                                })
        self._maybe_raise(response)

        result = response.json().get('result', {})
        cubegen_id = result.get('cubegen_id')

        with observe_progress('Generating cube', 100) as cm:
            while True:
                time.sleep(1.0)

                response = requests.get(self.endpoint_op(f'cubegens/{cubegen_id}'),
                                        json=self._gen_config,
                                        headers={
                                            **BASE_HEADERS,
                                            'Authorization': f'Bearer {self.access_token}',
                                        })
                self._maybe_raise(response)

                result = response.json().get('result')
                print(result)
                status = result.get('status')
                active = status.get('active')
                failed = status.get('failed')
                succeeded = result.get('succeeded')
                if succeeded:
                    return
                if failed:
                    raise CubeGeneratorError('Failed to generate cube')

                # TODO: turn response into progress
                cm.worked(1)
                if cm.state.progress >= 1.0:
                    return

    @classmethod
    def _maybe_raise(cls, response):
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise CubeGeneratorError(f'{e}') from e
