import os
from typing import Optional, Dict, Mapping

import yaml

from xcube.util.undefined import UNDEFINED
from xcube.webapi.context import ServiceContext, MultiLevelDatasetOpener
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.reqparams import RequestParams


def new_test_service_context(config_file_name: str = 'config.yml',
                             ml_dataset_openers: Dict[str, MultiLevelDatasetOpener] = None,
                             prefix: str = None) -> ServiceContext:
    ctx = ServiceContext(base_dir=get_res_test_dir(), ml_dataset_openers=ml_dataset_openers, prefix=prefix)
    config_file = os.path.join(ctx.base_dir, config_file_name)
    with open(config_file, encoding='utf-8') as fp:
        ctx.config = yaml.safe_load(fp)
    return ctx


def get_res_test_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), 'res', 'test'))


def get_res_demo_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'xcube', 'webapi', 'res', 'demo'))


class RequestParamsMock(RequestParams):
    def __init__(self, **kvp):
        self.kvp = kvp

    def get_query_arguments(self) -> Mapping[str, str]:
        return dict(self.kvp)

    def get_query_argument(self, name: str, default: Optional[str] = UNDEFINED) -> Optional[str]:
        value = self.kvp.get(name, default)
        if value is UNDEFINED:
            raise ServiceBadRequestError(f'Missing query parameter "{name}"')
        return value
