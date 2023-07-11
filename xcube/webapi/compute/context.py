# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
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

from xcube.server.api import Context
from xcube.webapi.common.context import ResourcesContext
from xcube.webapi.datasets.context import DatasetsContext
from xcube.webapi.places import PlacesContext


class ComputeContext(ResourcesContext):

    def __init__(self, server_ctx: Context):
        super().__init__(server_ctx)
        self._datasets_ctx: DatasetsContext \
            = server_ctx.get_api_ctx("datasets")
        assert isinstance(self._datasets_ctx, DatasetsContext)
        self._places_ctx: PlacesContext \
            = server_ctx.get_api_ctx("places")
        assert isinstance(self._places_ctx, PlacesContext)

    @property
    def datasets_ctx(self) -> DatasetsContext:
        return self._datasets_ctx

    @property
    def places_ctx(self) -> PlacesContext:
        return self._places_ctx
