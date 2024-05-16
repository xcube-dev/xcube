from typing import Mapping, Dict

from xcube.constants import LOG
from xcube.util.perf import measure_time_cm
from .context import StatisticsContext


def compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name: str,
    params: Mapping[str, str],
):
    params = dict(params)
    trace_perf = params.pop("debug", "1" if ctx.datasets_ctx.trace_perf else "0") == "1"
    measure_time = measure_time_cm(logger=LOG, disabled=not trace_perf)
    with measure_time("Computing statistics"):
        return _compute_statistics(ctx, ds_id, var_name, params)


def _compute_statistics(
    ctx: StatisticsContext,
    ds_id: str,
    var_name: str,
    params: Dict[str, str],
):
    # TODO: implement me
    return {}
