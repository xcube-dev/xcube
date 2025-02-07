# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from typing import Dict, Optional, TypeVar


def caseless_dict(*args, **kwargs) -> dict:
    """Create a dictionary that compares its string keys in a case-insensitive manner."""
    return _CaselessDict(*args, **kwargs)


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)


class _CaselessDict(dict):
    def __init__(self, *args, **kwargs: _VT):
        super().__init__(*args, **kwargs)
        self._lc_keys = {k.lower() if isinstance(k, str) else k: k for k in self.keys()}

    def __getitem__(self, k: _KT) -> _VT:
        if isinstance(k, str):
            k = self._lc_keys.get(k.lower(), k)
        return super().__getitem__(k)

    def __setitem__(self, k: _KT, v: _VT) -> None:
        if isinstance(k, str):
            lc_key = k.lower()
            if lc_key in self._lc_keys:
                k = self._lc_keys[lc_key]
            else:
                self._lc_keys[lc_key] = k
        super().__setitem__(k, v)

    def __delitem__(self, k: _KT) -> None:
        if isinstance(k, str):
            lc_key = k.lower()
            if lc_key in self._lc_keys:
                k = self._lc_keys.pop(lc_key)
        super().__delitem__(k)

    def __contains__(self, k: object) -> bool:
        if isinstance(k, str):
            return k.lower() in self._lc_keys
        return super().__contains__(k)

    def get(self, k: _KT, default: _VT = None) -> Optional[_VT_co]:
        if isinstance(k, str):
            lc_key = k.lower()
            if lc_key in self._lc_keys:
                k = self._lc_keys[lc_key]
            else:
                return default
        return super().get(k, default) if default is not None else super().get(k)

    def pop(self, k: _KT) -> _VT:
        if isinstance(k, str):
            lc_key = k.lower()
            if lc_key in self._lc_keys:
                k = self._lc_keys.pop(lc_key)
        return super().pop(k)
