# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import importlib
import os.path
import sys
import uuid
from typing import Sequence, Any, Dict, Callable, Mapping

import xarray as xr

from xcube.constants import LOG
from xcube.util.perf import measure_time
from .abc import MultiLevelDataset
from .lazy import LazyMultiLevelDataset

COMPUTE_DATASET = 'compute_dataset'


# TODO (forman): rename to ScriptedMultiLevelDataset
# TODO (forman): use new xcube.core.byoa package here

class ComputedMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset whose level datasets are computed
    by a user function.

    The script can import other Python modules located in the same
    directory as *script_path*.
    """

    def __init__(self,
                 script_path: str,
                 callable_name: str,
                 input_ml_dataset_ids: Sequence[str],
                 input_ml_dataset_getter: Callable[[str], MultiLevelDataset],
                 input_parameters: Mapping[str, Any],
                 ds_id: str = None,
                 exception_type: type = ValueError):

        input_parameters = input_parameters or {}
        super().__init__(ds_id=ds_id, parameters=input_parameters)

        # Allow scripts to import modules from within directory
        script_parent = os.path.dirname(script_path)
        if script_parent not in sys.path:
            sys.path = [script_parent] + sys.path
            LOG.info(f'Python sys.path prepended by {script_parent}')

        module_name, ext = os.path.splitext(os.path.basename(script_path))
        if not (ext == '.py' or ext == ""):
            LOG.warning(f"Unrecognized Python module file extension {ext!r}")

        if not callable_name.isidentifier():
            raise exception_type(
                f"Invalid dataset descriptor {ds_id!r}:"
                f" {callable_name!r} is not a valid Python identifier"
            )

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}: {e}"
            ) from e

        callable_obj = getattr(module, callable_name, None)
        if callable_obj is None:
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}:"
                f" no callable named {callable_name!r}"
                f" found in {script_path!r}"
            )
        if not callable(callable_obj):
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}:"
                f" object {callable_name!r} in {script_path!r}"
                f" is not callable"
            )

        if not callable_name or not callable_name.isidentifier():
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}:"
                f" {callable_name!r} is not a valid Python identifier"
            )
        if not input_ml_dataset_ids:
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}:"
                f" Input dataset(s) missing for callable {callable_name!r}"
            )
        for input_param_name in input_parameters.keys():
            if not input_param_name or not input_param_name.isidentifier():
                raise exception_type(
                    f"Invalid in-memory dataset descriptor {ds_id!r}: "
                    f"Input parameter {input_param_name!r}"
                    f" for callable {callable_name!r} "
                    f"is not a valid Python identifier"
                )
        LOG.info(f'Imported {callable_name}() from {script_path}')
        self._callable_name = callable_name
        self._callable_obj = callable_obj
        self._input_ml_dataset_ids = input_ml_dataset_ids
        self._input_ml_dataset_getter = input_ml_dataset_getter
        self._exception_type = exception_type

    def _get_num_levels_lazily(self) -> int:
        ds_0 = self._input_ml_dataset_getter(self._input_ml_dataset_ids[0])
        return ds_0.num_levels

    def _get_dataset_lazily(self, index: int,
                            parameters: Dict[str, Any]) -> xr.Dataset:
        input_datasets = [
            self._input_ml_dataset_getter(ds_id).get_dataset(index)
            for ds_id in self._input_ml_dataset_ids]
        try:
            with measure_time(tag=f"Computed in-memory dataset"
                                  f" {self.ds_id!r} at level {index}"):
                computed_value = self._callable_obj(*input_datasets,
                                                    **parameters)
        except Exception as e:
            raise self._exception_type(
                f"Failed to compute in-memory dataset {self.ds_id!r}"
                f" at level {index} "
                f"from function {self._callable_name!r}: {e}"
            ) from e
        if not isinstance(computed_value, xr.Dataset):
            raise self._exception_type(
                f"Failed to compute in-memory dataset {self.ds_id!r}"
                f" at level {index} "
                f"from function {self._callable_name!r}: "
                f"expected an xarray.Dataset but got {type(computed_value)}"
            )
        return computed_value


def open_ml_dataset_from_python_code(
        script_path: str,
        callable_name: str,
        input_ml_dataset_ids: Sequence[str] = None,
        input_ml_dataset_getter: Callable[[str], MultiLevelDataset] = None,
        input_parameters: Mapping[str, Any] = None,
        ds_id: str = None,
        exception_type: type = ValueError
) -> MultiLevelDataset:
    with measure_time(tag=f"Opened memory dataset {script_path}"):
        return ComputedMultiLevelDataset(script_path,
                                         callable_name,
                                         input_ml_dataset_ids,
                                         input_ml_dataset_getter,
                                         input_parameters,
                                         ds_id=ds_id,
                                         exception_type=exception_type)


def augment_ml_dataset(
        ml_dataset: MultiLevelDataset,
        script_path: str,
        callable_name: str,
        input_ml_dataset_getter: Callable[[str], MultiLevelDataset],
        input_ml_dataset_setter: Callable[[MultiLevelDataset], None],
        input_parameters: Mapping[str, Any] = None,
        exception_type: type = ValueError
):
    from .identity import IdentityMultiLevelDataset
    from .combined import CombinedMultiLevelDataset
    with measure_time(tag=f"Added augmentation from {script_path}"):
        orig_id = ml_dataset.ds_id
        aug_id = uuid.uuid4()
        aug_inp_id = f'aug-input-{aug_id}'
        aug_inp_ds = IdentityMultiLevelDataset(ml_dataset, ds_id=aug_inp_id)
        input_ml_dataset_setter(aug_inp_ds)
        aug_ds = ComputedMultiLevelDataset(script_path,
                                           callable_name,
                                           [aug_inp_id],
                                           input_ml_dataset_getter,
                                           input_parameters,
                                           ds_id=f'aug-{aug_id}',
                                           exception_type=exception_type)
        return CombinedMultiLevelDataset([ml_dataset, aug_ds], ds_id=orig_id)
