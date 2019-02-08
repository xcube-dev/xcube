import glob
import os
from typing import Sequence


def get_inputdata_file(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'inputdata', name)


def get_inputdata_files(pattern: str) -> Sequence[str]:
    return glob.glob(os.path.join(os.path.dirname(__file__), 'inputdata', pattern))
