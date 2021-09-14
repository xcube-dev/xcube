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

# noinspection PyUnresolvedReferences
from xcube.core.byoa import CodeConfig
# noinspection PyUnresolvedReferences
from xcube.core.byoa import FileSet
# noinspection PyUnresolvedReferences
from xcube.core.store.descriptor import DatasetDescriptor

from .config import CallbackConfig
from .config import CubeConfig
from .config import InputConfig
from .config import OutputConfig
from .error import CubeGeneratorError
from .generator import CubeGenerator
from .local.generator import LocalCubeGenerator
from .processor import DatasetProcessor
from .processor import METHOD_NAME_DATASET_PROCESSOR
from .processor import METHOD_NAME_PARAMS_SCHEMA_GETTER
from .remote.config import ServiceConfig
from .remote.config import ServiceConfigLike
from .remote.generator import RemoteCubeGenerator
from .remote.response import CostEstimation
from .remote.response import CubeInfoWithCosts
from .remote.response import CubeInfoWithCostsResult
from .request import CubeGeneratorRequest
from .request import CubeGeneratorRequestLike
from .response import CubeGeneratorResult
from .response import CubeInfo
from .response import CubeInfoResult
from .response import CubeReference
