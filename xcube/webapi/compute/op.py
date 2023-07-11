from typing import Callable, Dict

_OP_REGISTRY: Dict[str, Callable] = {}


def get_operations() -> Dict[str, Callable]:
    return _OP_REGISTRY.copy()


def op(op_id: str = None):
    def decorator(func):
        reg_op_id = op_id or func.__name__
        print(f"registered {reg_op_id}: {func}")
        _OP_REGISTRY[reg_op_id] = func
        return func

    return decorator
