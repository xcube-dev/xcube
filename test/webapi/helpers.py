import os
from typing import Optional

import yaml

from xcube.webapi.context import ServiceContext
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.reqparams import RequestParams
from xcube.util.undefined import UNDEFINED


def new_test_service_context() -> ServiceContext:
    ctx = ServiceContext(base_dir=get_res_test_dir())
    config_file = os.path.join(ctx.base_dir, 'config.yml')
    with open(config_file) as fp:
        ctx.config = yaml.safe_load(fp)
    return ctx


def new_demo_service_context() -> ServiceContext:
    ctx = ServiceContext(base_dir=get_res_demo_dir())
    config_file = os.path.join(ctx.base_dir, 'config.yml')
    with open(config_file) as fp:
        ctx.config = yaml.safe_load(fp)
    return ctx


def get_res_test_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), 'res', 'test'))


def get_res_demo_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'xcube', 'webapi', 'res', 'demo'))


class RequestParamsMock(RequestParams):
    def __init__(self, **kvp):
        self.kvp = kvp

    def get_query_argument(self, name: str, default: Optional[str] = UNDEFINED) -> Optional[str]:
        value = self.kvp.get(name, default)
        if value is UNDEFINED:
            raise ServiceBadRequestError(f'Missing query parameter "{name}"')
        return value
