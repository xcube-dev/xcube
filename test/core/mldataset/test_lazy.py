# Copyright (c) 2018-2026 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import threading
import unittest
from typing import Any
from unittest import mock

import xarray as xr

from xcube.core.mldataset.lazy import LazyMultiLevelDataset


class _ObservedRLock:
    """Lock wrapper that makes the race deterministic for tests."""

    def __init__(self):
        self._lock = threading.RLock()
        self.first_enter_started = threading.Event()
        self.finish_first_enter = threading.Event()
        self.wait_attempted = threading.Event()
        self.enter_count = 0

    def __enter__(self):
        # Record when a second thread has to wait for the first thread.
        if not self._lock.acquire(blocking=False):
            self.wait_attempted.set()
            self._lock.acquire()
        self.enter_count += 1
        if self.enter_count == 1:
            # Hold the first thread inside the lock until the test releases it.
            self.first_enter_started.set()
            if not self.finish_first_enter.wait(timeout=5):
                raise TimeoutError("timed out waiting to finish first lock entry")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


class _ConcurrentLazyMultiLevelDataset(LazyMultiLevelDataset):
    """Minimal lazy dataset that counts lazy loads."""

    def __init__(self):
        super().__init__()
        self._lock = _ObservedRLock()
        self.grid_mapping_load_count = 0
        self.num_levels_load_count = 0
        self.dataset_load_count = 0

    def _get_num_levels_lazily(self) -> int:
        self.num_levels_load_count += 1
        return 1

    def _get_dataset_lazily(self, index: int, parameters: dict[str, Any]) -> xr.Dataset:
        self.dataset_load_count += 1
        return xr.Dataset(attrs={"load_count": self.dataset_load_count})

    def _get_grid_mapping_lazily(self):
        self.grid_mapping_load_count += 1
        return object()


class LazyMultiLevelDatasetTest(unittest.TestCase):
    def test_ds_id_computes_missing_value_only_once_if_threads_race(self):
        ml_dataset = _ConcurrentLazyMultiLevelDataset()

        # If the race is not prevented, the second generated ID will be used.
        with mock.patch(
            "xcube.core.mldataset.lazy.uuid.uuid4",
            side_effect=["first-id", "second-id"],
        ) as uuid4:
            results = self._run_two_threads(
                lambda: ml_dataset.ds_id, ml_dataset.lock
            )

        self.assertEqual(["first-id", "first-id"], results)
        # The inner lock check must prevent the waiting thread from
        # generating a new ID.
        self.assertEqual(1, uuid4.call_count)

    def test_grid_mapping_computes_missing_value_only_once_if_threads_race(self):
        ml_dataset = _ConcurrentLazyMultiLevelDataset()

        results = self._run_two_threads(
            lambda: ml_dataset.grid_mapping, ml_dataset.lock
        )

        self.assertEqual(1, ml_dataset.grid_mapping_load_count)
        self.assertIs(results[0], results[1])

    def test_num_levels_computes_missing_value_only_once_if_threads_race(self):
        ml_dataset = _ConcurrentLazyMultiLevelDataset()

        results = self._run_two_threads(
            lambda: ml_dataset.num_levels, ml_dataset.lock
        )

        self.assertEqual([1, 1], results)
        self.assertEqual(1, ml_dataset.num_levels_load_count)

    def test_get_dataset_computes_missing_level_only_once_if_threads_race(self):
        ml_dataset = _ConcurrentLazyMultiLevelDataset()

        results = self._run_two_threads(
            lambda: ml_dataset.get_dataset(0), ml_dataset.lock
        )

        self.assertEqual(1, ml_dataset.dataset_load_count)
        self.assertIs(results[0], results[1])

    def _run_two_threads(self, getter, lock: _ObservedRLock):
        results = [None, None]
        errors = []

        def load_value(slot: int):
            try:
                results[slot] = getter()
            except BaseException as e:
                errors.append(e)

        # Thread 1 enters the lock first and is paused there by _ObservedRLock.
        thread_1 = threading.Thread(target=load_value, args=(0,))
        thread_2 = None
        thread_1.start()

        try:
            self.assertTrue(lock.first_enter_started.wait(timeout=5))

            # Thread 2 must now queue behind thread 1, reproducing the race.
            thread_2 = threading.Thread(target=load_value, args=(1,))
            thread_2.start()
            self.assertTrue(lock.wait_attempted.wait(timeout=5))
        finally:
            # Always release and join threads, even if an assertion fails.
            lock.finish_first_enter.set()
            thread_1.join(timeout=5)
            if thread_2 is not None:
                thread_2.join(timeout=5)

        self.assertFalse(thread_1.is_alive())
        self.assertIsNotNone(thread_2)
        self.assertFalse(thread_2.is_alive())
        if errors:
            raise errors[0]
        return results
