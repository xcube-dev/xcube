# The MIT License (MIT)
# Copyright (c) 2022 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import threading
from typing import Optional, Union

import tornado.ioloop
import xarray as xr

from xcube.core.mldataset import MultiLevelDataset
from xcube.server.api import ServerConfig
from xcube.server.server import Server
from xcube.server.webservers.tornado import TornadoFramework
from xcube.webapi.datasets.context import DatasetsContext


class InteractiveServer:
    def __init__(self):
        self._io_loop = tornado.ioloop.IOLoop()
        thread = threading.Thread(target=self._io_loop.start)
        thread.daemon = True
        thread.start()

        config = {
            "port": 8080,
            "address": "0.0.0.0",
            "static_routes": [
                ["/viewer", "D:\\Projects\\xcube-viewer\\build"],
            ]
        }

        self._server = Server(TornadoFramework(io_loop=self._io_loop),
                              config=config)

        self._io_loop.add_callback(self._server.start)

    def stop(self):
        if self._server is not None:
            # noinspection PyBroadException
            try:
                self._server.stop()
            except:
                pass
        self._server = None
        self._io_loop = None

    def is_running(self) -> bool:
        return self._server is not None

    def add_dataset(self,
                    dataset: Union[xr.Dataset, MultiLevelDataset],
                    ds_id: Optional[str] = None,
                    title: Optional[str] = None):
        if not self._check_running():
            return
        datasets_ctx: DatasetsContext = \
            self._server.ctx.get_api_ctx('datasets')
        return datasets_ctx.add_dataset(dataset, ds_id=ds_id, title=title)

    def remove_dataset(self, ds_id: str):
        if not self._check_running():
            return
        datasets_ctx: DatasetsContext = \
            self._server.ctx.get_api_ctx('datasets')
        datasets_ctx.remove_dataset(ds_id)

    def _check_running(self):
        is_running = self.is_running()
        if not is_running:
            print('Server not running')
        return is_running
