# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import os.path
import uuid
from typing import Any, Dict, Callable, Optional, Tuple
from collections.abc import Sequence, Mapping

import xarray as xr

from xcube.core.byoa import CodeConfig
from xcube.core.byoa import FileSet
from xcube.core.gridmapping import GridMapping
from xcube.util.assertions import assert_given
from xcube.util.assertions import assert_instance
from xcube.util.assertions import assert_true
from xcube.util.perf import measure_time
from .abc import MultiLevelDataset
from .lazy import LazyMultiLevelDataset

MultiLevelDatasetGetter = Callable[[str], MultiLevelDataset]
MultiLevelDatasetSetter = Callable[[MultiLevelDataset], None]


class ComputedMultiLevelDataset(LazyMultiLevelDataset):
    """A multi-level dataset whose level datasets are computed
    by a user function.

    The script can import other Python modules located in the same
    directory as *script_path*.
    """

    def __init__(
        self,
        script_path: str,
        callable_name: str,
        input_ml_dataset_ids: Sequence[str],
        input_ml_dataset_getter: MultiLevelDatasetGetter,
        input_parameters: Optional[Mapping[str, Any]] = None,
        ds_id: str = "",
        exception_type: type = ValueError,
    ):
        callable_ref, callable_obj = self.get_callable(
            script_path,
            callable_name,
            input_ml_dataset_ids,
            input_ml_dataset_getter,
            input_parameters=input_parameters,
            ds_id=ds_id,
            exception_type=exception_type,
        )

        super().__init__(ds_id=ds_id, parameters=input_parameters)
        self._callable_ref = callable_ref
        self._callable_obj = callable_obj
        self._input_ml_dataset_ids = input_ml_dataset_ids
        self._input_ml_dataset_getter = input_ml_dataset_getter
        self._exception_type = exception_type

    @classmethod
    def get_callable(
        cls,
        script_path: str,
        callable_name: str,
        input_ml_dataset_ids: Sequence[str],
        input_ml_dataset_getter: MultiLevelDatasetGetter,
        input_parameters: Optional[Mapping[str, Any]] = None,
        ds_id: str = "",
        exception_type: type = ValueError,
    ) -> tuple[str, Callable]:
        assert_instance(script_path, str, name="script_path")
        assert_given(script_path, name="script_path")
        assert_true(
            callable(input_ml_dataset_getter),
            message=f"input_ml_dataset_getter must be a callable",
        )
        assert_given(input_ml_dataset_getter, name="input_ml_dataset_getter")
        assert_instance(ds_id, str, name="ds_id")
        assert_given(ds_id, name="ds_id")

        module_name = None
        basename = os.path.basename(script_path)
        basename, ext = os.path.splitext(basename)
        if ext == ".py":
            script_path = os.path.dirname(script_path)
            module_name = basename

        if ":" in callable_name:
            callable_ref = callable_name
        else:
            if not module_name:
                raise exception_type(
                    f"Invalid in-memory dataset descriptor {ds_id!r}:"
                    f" Missing module name in {callable_name!r}"
                )
            callable_ref = f"{module_name}:{callable_name}"

        if not input_ml_dataset_ids:
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}:"
                f" Input dataset(s) missing for callable {callable_name!r}"
            )

        for input_param_name in (input_parameters or {}).keys():
            if not input_param_name or not input_param_name.isidentifier():
                raise exception_type(
                    f"Invalid in-memory dataset descriptor {ds_id!r}:"
                    f" Input parameter {input_param_name!r}"
                    f" for callable {callable_name!r}"
                    f" is not a valid Python identifier"
                )

        try:
            callable_obj = CodeConfig.from_file_set(
                FileSet(path=script_path),
                callable_ref=callable_ref,
                install_required=False,
            ).get_callable()
        except (TypeError, ValueError, ImportError) as e:
            raise exception_type(f"Invalid dataset descriptor {ds_id!r}: {e}") from e

        return callable_ref, callable_obj

    @property
    def num_inputs(self) -> int:
        return len(self._input_ml_dataset_ids)

    def get_input_dataset(self, index: int) -> MultiLevelDataset:
        return self._input_ml_dataset_getter(self._input_ml_dataset_ids[index])

    def _get_num_levels_lazily(self) -> int:
        return self.get_input_dataset(0).num_levels

    def _get_grid_mapping_lazily(self) -> GridMapping:
        return self.get_input_dataset(0).grid_mapping

    def _get_dataset_lazily(self, index: int, parameters: dict[str, Any]) -> xr.Dataset:
        input_datasets = [
            self._input_ml_dataset_getter(ds_id).get_dataset(index)
            for ds_id in self._input_ml_dataset_ids
        ]
        try:
            with measure_time(
                f"Computed in-memory dataset" f" {self.ds_id!r} at level {index}"
            ):
                computed_value = self._callable_obj(*input_datasets, **parameters)
        except Exception as e:
            raise self._exception_type(
                f"Failed to compute in-memory dataset {self.ds_id!r}"
                f" at level {index} "
                f"from function {self._callable_ref!r}(): {e}"
            ) from e
        if not isinstance(computed_value, xr.Dataset):
            raise self._exception_type(
                f"Failed to compute in-memory dataset {self.ds_id!r}"
                f" at level {index} "
                f"from function {self._callable_ref!r}(): "
                f"expected an xarray.Dataset but got {type(computed_value)}"
            )
        return computed_value


def augment_ml_dataset(
    ml_dataset: MultiLevelDataset,
    script_path: str,
    callable_name: str,
    input_ml_dataset_getter: MultiLevelDatasetGetter,
    input_ml_dataset_setter: MultiLevelDatasetSetter,
    input_parameters: Optional[Mapping[str, Any]] = None,
    is_factory: bool = False,
    exception_type: type = ValueError,
):
    from .identity import IdentityMultiLevelDataset
    from .combined import CombinedMultiLevelDataset

    with measure_time(f"Added augmentation from {script_path}"):
        orig_id = ml_dataset.ds_id
        aug_id = uuid.uuid4()
        aug_inp_id = f"aug-input-{aug_id}"
        aug_inp_ds = IdentityMultiLevelDataset(ml_dataset, ds_id=aug_inp_id)
        input_ml_dataset_setter(aug_inp_ds)
        aug_ds = _open_ml_dataset_from_python_code(
            script_path,
            callable_name,
            [aug_inp_id],
            input_ml_dataset_getter,
            input_parameters=input_parameters,
            is_factory=is_factory,
            ds_id=f"aug-{aug_id}",
            exception_type=exception_type,
        )
        return CombinedMultiLevelDataset([ml_dataset, aug_ds], ds_id=orig_id)


def open_ml_dataset_from_python_code(
    script_path: str,
    callable_name: str,
    input_ml_dataset_ids: Sequence[str],
    input_ml_dataset_getter: MultiLevelDatasetGetter,
    input_parameters: Optional[Mapping[str, Any]] = None,
    is_factory: bool = False,
    ds_id: str = "",
    exception_type: type = ValueError,
) -> MultiLevelDataset:
    with measure_time(f"Opened memory dataset {script_path}"):
        return _open_ml_dataset_from_python_code(
            script_path,
            callable_name,
            input_ml_dataset_ids,
            input_ml_dataset_getter,
            input_parameters=input_parameters,
            is_factory=is_factory,
            ds_id=ds_id,
            exception_type=exception_type,
        )


def _open_ml_dataset_from_python_code(
    script_path: str,
    callable_name: str,
    input_ml_dataset_ids: Sequence[str],
    input_ml_dataset_getter: MultiLevelDatasetGetter,
    input_parameters: Optional[Mapping[str, Any]] = None,
    is_factory: bool = False,
    ds_id: str = "",
    exception_type: type = ValueError,
) -> MultiLevelDataset:
    if is_factory:
        callable_ref, callable_obj = ComputedMultiLevelDataset.get_callable(
            script_path,
            callable_name,
            input_ml_dataset_ids,
            input_ml_dataset_getter,
            input_parameters=input_parameters,
            ds_id=ds_id,
            exception_type=exception_type,
        )
        input_datasets = [
            input_ml_dataset_getter(ds_id) for ds_id in input_ml_dataset_ids
        ]
        try:
            ml_dataset = callable_obj(*input_datasets, **(input_parameters or {}))
            if not isinstance(ml_dataset, MultiLevelDataset):
                raise TypeError(
                    f"{callable_ref!r} must return instance of"
                    f" xcube.core.mldataset.MultiLevelDataset,"
                    f" but was {type(ml_dataset)}"
                )
            ml_dataset.ds_id = ds_id
            return ml_dataset
        except BaseException as e:
            raise exception_type(
                f"Invalid in-memory dataset descriptor {ds_id!r}: {e}"
            ) from e
    else:
        return ComputedMultiLevelDataset(
            script_path,
            callable_name,
            input_ml_dataset_ids,
            input_ml_dataset_getter,
            input_parameters=input_parameters,
            ds_id=ds_id,
            exception_type=exception_type,
        )
