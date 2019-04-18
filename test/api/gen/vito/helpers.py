import os


def get_inputdata_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)
