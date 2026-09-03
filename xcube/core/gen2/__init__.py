# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.core.byoa import CodeConfig, FileSet
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

__all__ = [
    "CallbackConfig",
    "CodeConfig",
    "CostEstimation",
    "CubeConfig",
    "CubeGenerator",
    "CubeGeneratorError",
    "CubeGeneratorRequest",
    "CubeGeneratorRequestLike",
    "CubeGeneratorResult",
    "CubeInfo",
    "CubeInfoResult",
    "CubeInfoWithCosts",
    "CubeInfoWithCostsResult",
    "CubeReference",
    "DatasetDescriptor",
    "DatasetProcessor",
    "FileSet",
    "InputConfig",
    "LocalCubeGenerator",
    "METHOD_NAME_DATASET_PROCESSOR",
    "METHOD_NAME_PARAMS_SCHEMA_GETTER",
    "OutputConfig",
    "RemoteCubeGenerator",
    "ServiceConfig",
    "ServiceConfigLike",
]
