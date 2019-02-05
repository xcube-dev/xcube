from abc import ABCMeta, abstractmethod
from io import IOBase
import os
import shutil


class FileSystem(metaclass=ABCMeta):
    @abstractmethod
    def open(self, path: str, mode='r') -> IOBase:
        pass

    @abstractmethod
    def mkdir(self, path: str):
        pass

    @abstractmethod
    def mkdirs(self, path: str):
        pass

    @abstractmethod
    def rename(self, from_path: str, to_path: str):
        pass

    @abstractmethod
    def copy(self, from_path: str, to_path: str):
        pass

    @abstractmethod
    def rm(self, path: str, recursive=False):
        pass


class LocalFileSystem(FileSystem):
    def open(self, path: str, mode='r') -> IOBase:
        # noinspection PyTypeChecker
        return open(path, mode=mode)

    def mkdir(self, path: str):
        os.mkdir(path)

    def mkdirs(self, path: str):
        os.makedirs(path)

    def rename(self, from_path: str, to_path: str):
        os.rename(from_path, to_path)

    def copy(self, from_path: str, to_path: str):
        shutil.copy(from_path, to_path)

    def rm(self, path: str, recursive=False):
        raise NotImplementedError()


class MemoryFileSystem(FileSystem):
    def __init__(self):
        self.root = dict()

    def open(self, path: str, mode='r') -> IOBase:
        raise NotImplementedError()

    def mkdir(self, path: str):
        raise NotImplementedError()

    def mkdirs(self, path: str):
        raise NotImplementedError()

    def rename(self, from_path: str, to_path: str):
        raise NotImplementedError()

    def copy(self, from_path: str, to_path: str):
        raise NotImplementedError()

    def rm(self, path: str, recursive=False):
        raise NotImplementedError()
