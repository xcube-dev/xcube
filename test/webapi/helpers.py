import os
from typing import Dict, Mapping, Union
from typing import Optional, Type, TypeVar

import yaml

from test.server.mocks import MockFramework
from xcube.server.api import Context
from xcube.server.api import ServerConfig
from xcube.server.framework import Framework
from xcube.server.server import Server
from xcube.util.extension import ExtensionRegistry
from xcube.util.plugin import get_extension_registry
from xcube.util.undefined import UNDEFINED
from xcube.webapi.context import ServiceContext, MultiLevelDatasetOpener
from xcube.webapi.errors import ServiceBadRequestError
from xcube.webapi.reqparams import RequestParams

XCUBE_TEST_CLIENT_ID = os.environ.get('XCUBE_TEST_CLIENT_ID')
XCUBE_TEST_CLIENT_SECRET = os.environ.get('XCUBE_TEST_CLIENT_SECRET')

T = TypeVar('T', bound=Context)


def get_api_ctx(api_name: str,
                api_ctx_cls: Type[T],
                server_config: Union[str, ServerConfig],
                framework: Optional[Framework] = None,
                extension_registry: Optional[ExtensionRegistry] = None) -> T:
    """Get the API context object for the given
    API name *api_name*,
    API context class *api_ctx_cls*,
    and server configuration path or dictionary *server_config*.

    :param api_name: The name of the API, e.g. "auth"
    :param api_ctx_cls: The API context class
    :param server_config: Server configuration string or dictionary.
        If a relative path is passed it is made absolute using prefix
        "${project}/test/webapi/res/test".
    :param framework: Web framework, defaults to a MockFramework.
    :param extension_registry: Extension registry,
        defaults to xcube's populated default extension registry
    :return: The API context object
    :raise AssertionError: if API context object can not be determined
    """
    if isinstance(server_config, str):
        config_path = server_config
        base_dir = os.path.dirname(config_path)
        if not base_dir:
            base_dir = get_res_test_dir()
            config_path = os.path.join(base_dir, config_path)
        elif not os.path.isabs(base_dir):
            base_dir = os.path.abspath(base_dir)
        with open(config_path, encoding='utf-8') as fp:
            server_config = yaml.safe_load(fp)
            assert isinstance(server_config, dict)
            server_config["base_dir"] = base_dir
    else:
        assert isinstance(server_config, dict)

    framework = framework or MockFramework()
    extension_registry = extension_registry or get_extension_registry()
    server = Server(framework,
                    server_config,
                    extension_registry=extension_registry)
    api_ctx = server.ctx.get_api_ctx(api_name)
    assert isinstance(api_ctx, api_ctx_cls)
    return api_ctx


def new_test_service_context(config_file_name: str = 'config.yml',
                             ml_dataset_openers: Dict[
                                 str, MultiLevelDatasetOpener] = None,
                             prefix: str = None) -> ServiceContext:
    ctx = ServiceContext(base_dir=get_res_test_dir(),
                         ml_dataset_openers=ml_dataset_openers, prefix=prefix)
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
        if value == UNDEFINED:
            raise ServiceBadRequestError(f'Missing query parameter "{name}"')
        return value
