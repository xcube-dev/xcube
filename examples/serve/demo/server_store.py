from xcube.constants import (
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_ADDRESS,
)
from xcube.cli.common import configure_cli_output
from xcube.server.server import Server
from xcube.server.webservers.tornado import TornadoFramework
from xcube.util.config import load_configs
from pathlib import Path
import os

configure_cli_output(quiet=False, verbosity="-vvv")
_DIR = str(Path(__file__).resolve().parent)
config = load_configs(os.path.join(_DIR, "config.yml"))
port = DEFAULT_SERVER_PORT
config["port"] = port
address = DEFAULT_SERVER_ADDRESS
config["address"] = address

framework = TornadoFramework()
server = Server(framework, config)
server.start()
