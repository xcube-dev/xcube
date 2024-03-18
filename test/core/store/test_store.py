from fsspec.registry import register_implementation
from xcube.core.store import new_data_store, DataStoreError
import pytest


def test_fsspec_instantiation_error():
    error_string = "deliberate instantiation error for testing"
    register_implementation(
        "abfs", "nonexistentmodule.NonexistentClass", True, error_string
    )
    with pytest.raises(DataStoreError, match=error_string):
        new_data_store("abfs").list_data_ids()
