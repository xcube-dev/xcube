# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import datetime
import re
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from xcube.util.assertions import assert_in

REF_DATETIME_STR = "1970-01-01 00:00:00"
REF_DATETIME = pd.to_datetime(REF_DATETIME_STR, utc=True)
DATETIME_UNITS = f"days since {REF_DATETIME_STR}"
DATETIME_CALENDAR = "gregorian"
SECONDS_PER_DAY = 24 * 60 * 60
MICROSECONDS_PER_DAY = 1000 * 1000 * SECONDS_PER_DAY

_RE_TO_DATETIME_FORMATS = patterns = [
    (re.compile(14 * "\\d"), "%Y%m%d%H%M%S"),
    (re.compile(12 * "\\d"), "%Y%m%d%H%M"),
    (re.compile(8 * "\\d"), "%Y%m%d"),
    (re.compile(6 * "\\d"), "%Y%m"),
    (re.compile(4 * "\\d"), "%Y"),
]


def add_time_coords(dataset: xr.Dataset, time_range: tuple[float, float]) -> xr.Dataset:
    t1, t2 = time_range
    if t1 != t2:
        t_center = (t1 + t2) / 2
    else:
        t_center = t1
    dataset = dataset.expand_dims("time")
    dataset = dataset.assign_coords(
        time=(["time"], from_time_in_days_since_1970([t_center]))
    )
    time_var = dataset.coords["time"]
    time_var.attrs["long_name"] = "time"
    time_var.attrs["standard_name"] = "time"
    # Avoiding xarray error:
    #   ValueError: failed to prevent overwriting existing key units in attrs on variable 'time'.
    #   This is probably an encoding field used by xarray to describe how a variable is serialized.
    #   To proceed, remove this key from the variable's attributes manually.
    # time_var.attrs['units'] = DATETIME_UNITS
    # time_var.attrs['calendar'] = DATETIME_CALENDAR
    time_var.encoding["units"] = DATETIME_UNITS
    time_var.encoding["calendar"] = DATETIME_CALENDAR
    if t1 != t2:
        time_var.attrs["bounds"] = "time_bnds"
        dataset = dataset.assign_coords(
            time_bnds=(
                ["time", "bnds"],
                from_time_in_days_since_1970([t1, t2]).reshape(1, 2),
            )
        )
        time_bnds_var = dataset.coords["time_bnds"]
        time_bnds_var.attrs["long_name"] = "time"
        time_bnds_var.attrs["standard_name"] = "time"
        # Avoiding xarray error:
        #   ValueError: failed to prevent overwriting existing key units in attrs on variable
        #   'time'. This is probably an encoding field used by xarray to describe how a variable
        #   is serialized.
        # To proceed, remove this key from the variable's attributes manually.
        # time_bnds_var.attrs['units'] = DATETIME_UNITS
        # time_bnds_var.attrs['calendar'] = DATETIME_CALENDAR
        time_bnds_var.encoding["units"] = DATETIME_UNITS
        time_bnds_var.encoding["calendar"] = DATETIME_CALENDAR
    return dataset


def get_time_range_from_data(
    dataset: xr.Dataset, maybe_consider_metadata: bool = True
) -> tuple[Optional[float], Optional[float]]:
    """Determines a time range from a dataset by inspecting its time_bounds or time data arrays.
    In cases where no time bounds are given and no time periodicity can be determined,
    metadata may be considered.

    Args:
        dataset: The dataset of which the time range shall be determined
        maybe_consider_metadata: Whether metadata shall be considered.
    Only used when the dataset has no time bounds array and no time periodicity.
    The values will only be set when they do not contradict the values from the data arrays.

    Returns:
        A tuple with two float values: The first one represents the
        start time,
    the second the end time. Either may be None.
    """
    time_bounds_names = ["time_bnds", "time_bounds"]
    for time_bounds_name in time_bounds_names:
        if time_bounds_name in dataset:
            return _get_time_range_from_time_bounds(dataset, time_bounds_name)
    # handle special case with datasets that have special coordinates 'start_time' and 'end_time'.
    # This is the case for, e.g, ICESHEETS Greenland
    if "start_time" in dataset and "end_time" in dataset:
        return dataset["start_time"].values[0], dataset["end_time"].values[-1]
    time_names = ["time", "t"]
    time = None
    for time_name in time_names:
        if time_name in dataset:
            time = dataset[time_name]
            break
    if time is None:
        return None, None
    time_bnds_name = time.attrs.get("bounds", "time_bnds")
    if time_bnds_name in dataset:
        return _get_time_range_from_time_bounds(dataset, time_bnds_name)
    is_cf_time = isinstance(time[0].values.item(), cftime.datetime)
    data_start = (
        pd.to_datetime(time[0].values.item().isoformat())
        if is_cf_time
        else time[0].values
    )
    data_end = (
        pd.to_datetime(time[-1].values.item().isoformat())
        if is_cf_time
        else time[-1].values
    )
    if time.size < 3:
        return _maybe_return_time_range_from_metadata(
            dataset, data_start, data_end, maybe_consider_metadata
        )
    time_diff = time.diff(dim=time.dims[0]).values
    time_res = time_diff[0]
    time_regular = all([time_res - diff == np.timedelta64(0) for diff in time_diff[1:]])
    if time_regular:
        return data_start - time_res / 2, data_end + time_res / 2
    return _maybe_return_time_range_from_metadata(
        dataset, data_start, data_end, maybe_consider_metadata
    )


def _maybe_return_time_range_from_metadata(
    dataset: xr.Dataset,
    data_start_time: float,
    data_end_time: float,
    maybe_consider_metadata: bool,
) -> tuple[float, float]:
    if maybe_consider_metadata:
        attr_start_time, attr_end_time = get_time_range_from_attrs(dataset)
        attr_start_time = pd.to_datetime(attr_start_time, utc=True)
        attr_end_time = pd.to_datetime(attr_end_time, utc=True)
        if attr_start_time is not None and attr_end_time is not None:
            try:
                if attr_start_time < data_start_time and attr_end_time > data_end_time:
                    return (
                        attr_start_time.to_datetime64(),
                        attr_end_time.to_datetime64(),
                    )
            except TypeError:
                try:
                    if (
                        attr_start_time.to_datetime64() < data_start_time
                        and attr_end_time.to_datetime64() > data_end_time
                    ):
                        return (
                            attr_start_time.to_datetime64(),
                            attr_end_time.to_datetime64(),
                        )
                except TypeError:
                    # use time values from data
                    pass
    return data_start_time, data_end_time


def _get_time_range_from_time_bounds(
    dataset: xr.Dataset, time_bounds_name: str
) -> tuple[Optional[float], Optional[float]]:
    time_bnds = dataset[time_bounds_name]
    if len(time_bnds.shape) == 2 and time_bnds.shape[1] == 2:
        return time_bnds[0, 0].values, time_bnds[-1, 1].values


def get_time_range_from_attrs(
    dataset: xr.Dataset,
) -> tuple[Optional[str], Optional[str]]:
    return get_start_time_from_attrs(dataset), get_end_time_from_attrs(dataset)


def get_start_time_from_attrs(dataset: xr.Dataset) -> Optional[str]:
    return _get_attr(
        dataset, ["time_coverage_start", "time_start", "start_time", "start_date"]
    )


def get_end_time_from_attrs(dataset: xr.Dataset) -> Optional[str]:
    return _get_attr(
        dataset,
        [
            "time_coverage_end",
            "time_stop",
            "time_end",
            "stop_time",
            "end_time",
            "stop_date",
            "end_date",
        ],
    )


def _get_attr(dataset: xr.Dataset, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in dataset.attrs:
            return remove_time_part_from_isoformat(str(dataset.attrs[name]))


def remove_time_part_from_isoformat(datetime_str: str) -> str:
    date_length = 10  # for example len("2010-02-04") == 10
    if len(datetime_str) > date_length and datetime_str[date_length] in ("T", " "):
        return datetime_str[0:date_length]
    return datetime_str


def to_time_in_days_since_1970(time_str: str, pattern=None) -> float:
    date_time = pd.to_datetime(time_str, format=pattern, utc=True)
    timedelta = date_time - REF_DATETIME
    return (
        timedelta.days
        + timedelta.seconds / SECONDS_PER_DAY
        + timedelta.microseconds / MICROSECONDS_PER_DAY
    )


def from_time_in_days_since_1970(
    time_value: Union[float, Sequence[float]],
) -> np.ndarray:
    if isinstance(time_value, int) or isinstance(time_value, float):
        return (
            pd.to_datetime(time_value, utc=True, unit="D", origin="unix")
            .round(freq="ms")
            .to_datetime64()
        )
    else:
        return np.array(list(map(from_time_in_days_since_1970, time_value)))


def timestamp_to_iso_string(
    time: Union[np.datetime64, datetime.datetime],
    freq: str = "s",
    round_fn: str = "round",
):
    """Convert a UTC timestamp given as nanos, millis, seconds, etc. since 1970-01-01 00:00:00
    to an ISO-format string.

    Args:
        time: UTC timestamp given as time delta since 1970-01-01
            00:00:00 in the units given by the numpy datetime64 type, so
            it can be as nanos, millis, seconds since 1970-01-01
            00:00:00.
        freq: time rounding resolution. See pandas.Timestamp.round().
        round_fn: time rounding function. Defaults to
            pandas.Timestamp.round().

    Returns:
        ISO-format string.
    """
    # All times are UTC (Z = Zulu Time Zone = UTC)
    assert_in(round_fn, ("ceil", "floor", "round"), name="round_fn")
    timestamp = pd.Timestamp(time)
    return getattr(timestamp, round_fn)(freq).isoformat() + "Z"


def find_datetime_format(line: str) -> tuple[Optional[str], int, int]:
    for regex, time_format in _RE_TO_DATETIME_FORMATS:
        searcher = regex.search(line)
        if searcher:
            p1, p2 = searcher.span()
            return time_format, p1, p2
    return None, -1, -1


def get_timestamp_from_string(string: str) -> Optional[pd.Timestamp]:
    time_format, p1, p2 = find_datetime_format(string)
    if time_format:
        try:
            return pd.to_datetime(string[p1:p2], format=time_format)
        except ValueError:
            pass


def get_timestamps_from_string(string: str) -> (pd.Timestamp, pd.Timestamp):
    first_time = None
    second_time = None
    time_format, p1, p2 = find_datetime_format(string)
    try:
        if time_format:
            first_time = pd.to_datetime(string[p1:p2], format=time_format)
        string_rest = string[p2:]
        time_format, p1, p2 = find_datetime_format(string_rest)
        if time_format:
            second_time = pd.to_datetime(string_rest[p1:p2], format=time_format)
    except ValueError:
        pass
    return first_time, second_time
