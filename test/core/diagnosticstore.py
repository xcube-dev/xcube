import time
from collections.abc import MutableMapping
from types import MethodType
from typing import TypeVar, Iterator, List, Callable, Any, Tuple

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')
_T_co = TypeVar('_T_co')
_VT_co = TypeVar('_VT_co')


class DiagnosticStore(MutableMapping):

    def __init__(self,
                 delegate: MutableMapping,
                 observer: Callable[[int, float, str, List[Tuple[str, Any]]], None] = None):
        self._delegate = delegate
        self._observer = observer or logging_observer()
        self._counter = 0
        self._add_optional_method('listdir', ['path'])
        self._add_optional_method('rmdir', ['path'])
        self._add_optional_method('rename', ['from_path', 'to_path'])

    def _add_optional_method(self, method_name: str, arg_names: List[str]):
        if hasattr(self._delegate, method_name):
            def method(_self, *args) -> List[str]:
                return _self.call_and_notify(method_name, *[(arg_names[i], args[i]) for i in range(len(args))])

            setattr(self, method_name, MethodType(method, self))

    def call_and_notify(self, method_name: str, *args):
        method = getattr(self._delegate, method_name)

        t0 = time.perf_counter()
        result = method(*(arg[1] for arg in args))
        t1 = time.perf_counter()

        self._counter += 1
        self._observer(self._counter, t1 - t0, method_name, *args)

        return result

    def __contains__(self, k: _KT) -> bool:
        return self.call_and_notify('__contains__', ('k', k))

    def __setitem__(self, k: _KT, v: _VT) -> None:
        return self.call_and_notify('__setitem__', ('k', k), ('v', v))

    def __delitem__(self, k: _KT) -> None:
        return self.call_and_notify('__delitem__', ('k', k))

    def __getitem__(self, k: _KT) -> _VT_co:
        return self.call_and_notify('__getitem__', ('k', k))

    def __len__(self) -> int:
        return self.call_and_notify('__len__')

    def __iter__(self) -> Iterator[_T_co]:
        return self.call_and_notify('__iter__')


def logging_observer(logger_name=None, log_path=None, log_all=False):
    import logging

    observer_logger = logging.getLogger(logger_name or 'diagnosticstore')

    if log_all:
        logger = logging.getLogger()
    else:
        logger = observer_logger

    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path or 'diagnosticstore.log')
    handler.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s'))
    logger.addHandler(handler)

    def observer(counter, time_needed, method_name, *args):
        msg = f'call #{counter}: {method_name}('
        msg += ', '.join(map(lambda x: f'{x[0]}={repr(x[1])}', args))
        msg += f'), took {int(1000 * time_needed)} ms'
        observer_logger.info(msg)

    return observer
