# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
from xcube.core.byoa import CodeConfig, FileSet

# noinspection PyUnresolvedReferences
from xcube.core.store.descriptor import DatasetDescriptor

from .config import CallbackConfig, CubeConfig, InputConfig, OutputConfig
from .error import CubeGeneratorError
from .generator import CubeGenerator
from .local.generator import LocalCubeGenerator
from .processor import (
    METHOD_NAME_DATASET_PROCESSOR,
    METHOD_NAME_PARAMS_SCHEMA_GETTER,
    DatasetProcessor,
)
from .remote.config import ServiceConfig, ServiceConfigLike
from .remote.generator import RemoteCubeGenerator
from .remote.response import CostEstimation, CubeInfoWithCosts, CubeInfoWithCostsResult
from .request import CubeGeneratorRequest, CubeGeneratorRequestLike
from .response import CubeGeneratorResult, CubeInfo, CubeInfoResult, CubeReference
