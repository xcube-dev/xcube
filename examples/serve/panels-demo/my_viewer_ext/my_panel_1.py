import altair as alt
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.ops
from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, Plot, Select

from xcube.constants import CRS_CRS84
from xcube.core.geom import mask_dataset_by_geometry, normalize_geometry
from xcube.core.gridmapping import GridMapping
from xcube.webapi.viewer.contrib import Panel
from xcube.webapi.viewer.contrib import get_dataset
from xcube.server.api import Context


panel = Panel(__name__, title="2D Histogram")


# Number of bins in x and y directions.
# This results in columns of 4096 items.
# Vega Altair's maximum is 5000.
NUM_BINS_MAX = 64


@panel.layout(State(source="app", property="controlState.selectedDatasetId"))
def render_panel(ctx: Context, dataset_id: str | None = None) -> Component:
    dataset = get_dataset(ctx, dataset_id)

    plot = Plot(
        id="plot",
        chart=None,
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": 300,
            "gap": 6,
        },
    )

    var_names, var_name_1, var_name_2 = get_var_select_options(dataset)

    select_var_1 = Select(
        id="select_var_1", label="Variable 1", value=var_name_1, options=var_names
    )
    select_var_2 = Select(
        id="select_var_2", label="Variable 2", value=var_name_2, options=var_names
    )

    button = Button(id="button", text="Update", style={"maxWidth": 100})

    controls = Box(
        children=[select_var_1, select_var_2, button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "gap": 6,
            "paddingTop": 50,
        },
    )

    return Box(
        children=[plot, controls],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "width": "100%",
            "height": "100%",
            "gap": 6,
            "padding": 6,
        },
    )


# noinspection PyUnusedLocal
@panel.callback(
    State(source="app", property="controlState.selectedDatasetId"),
    State(source="app", property="controlState.selectedTimeLabel"),
    State(source="app", property="controlState.selectedPlaceGeometry"),
    State("select_var_1"),
    State("select_var_2"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    time_label: float | None = None,
    place_geometry: str | None = None,
    var_name_1: str | None = None,
    var_name_2: str | None = None,
    _clicked: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:
    dataset = get_dataset(ctx, dataset_id)
    if dataset is None or place_geometry is None or not var_name_1 or not var_name_2:
        print("panel disabled")
        return None

    if "time" in dataset.coords:
        if time_label:
            dataset = dataset.sel(time=pd.Timestamp(time_label[0:-1]), method="nearest")
        else:
            dataset = dataset.isel(time=-1)

    grid_mapping = GridMapping.from_dataset(dataset)
    place_geometry = normalize_geometry(place_geometry)
    if place_geometry is not None and not grid_mapping.crs.is_geographic:
        project = pyproj.Transformer.from_crs(
            CRS_CRS84, grid_mapping.crs, always_xy=True
        ).transform
        place_geometry = shapely.ops.transform(project, place_geometry)

    dataset = mask_dataset_by_geometry(dataset, place_geometry)
    if dataset is None:
        print("dataset is None after masking, invalid geometry?")
        return None

    var_1_data: np.ndarray = dataset[var_name_1].values.ravel()
    var_2_data: np.ndarray = dataset[var_name_2].values.ravel()
    var_1_range = [np.nanmin(var_1_data), np.nanmax(var_1_data)]
    var_2_range = [np.nanmin(var_2_data), np.nanmax(var_2_data)]
    num_bins = min(NUM_BINS_MAX, var_1_data.size)
    hist2d, x_edges, y_edges = np.histogram2d(
        var_1_data,
        var_2_data,
        bins=num_bins,
        range=np.array([var_1_range, var_2_range]),
        density=True,
    )
    # x and y 2D arrays with bin centers
    x, y = np.meshgrid(np.arange(num_bins), np.arange(num_bins)[::-1])
    # z = hist2d
    z = np.where(hist2d > 0.0, hist2d, np.nan)
    source = pd.DataFrame(
        {var_name_1: x.ravel(), var_name_2: y.ravel(), "z": z.ravel()}
    )
    # TODO: use edges or center coordinates as tick labels.
    x_centers = x_edges[0:-1] + np.diff(x_edges) / 2
    y_centers = y_edges[0:-1] + np.diff(y_edges) / 2
    # TODO: limit number of ticks on axes to, e.g., 10.
    # TODO: allow chart to be adjusted to available container (<div>) size.
    return (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(
                f"{var_name_1}:O",
                # scale=alt.Scale(bins=x_centers),
            ),
            y=alt.Y(
                f"{var_name_2}:O",
                # scale=alt.Scale(bins=y_centers),
            ),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="viridis"), title="Density"),
        )
    ).properties(width=320, height=320)


@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    Input(source="app", property="controlState.selectedPlaceGeometry"),
    Output("button", "disabled"),
)
def enable_button(
    _ctx: Context,
    dataset_id: str | None = None,
    place_geometry: str | None = None,
) -> bool:
    return not dataset_id or not place_geometry


@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    State("select_var_1", "value"),
    State("select_var_2", "value"),
    Output("select_var_1", "options"),
    Output("select_var_1", "value"),
    Output("select_var_2", "options"),
    Output("select_var_2", "value"),
)
def populate_selects(
    ctx: Context,
    dataset_id: str | None = None,
    var_name_1: str | None = None,
    var_name_2: str | None = None,
) -> tuple[list, str | None, list, str | None]:
    dataset = get_dataset(ctx, dataset_id)
    var_names, var_name_1, var_name_2 = get_var_select_options(
        dataset, var_name_1, var_name_2
    )
    return var_names, var_name_1, var_names, var_name_2


def get_var_select_options(
    dataset,
    var_name_1: str | None = None,
    var_name_2: str | None = None,
) -> tuple[list, str | None, str | None]:

    if dataset is not None:
        var_names = [
            var_name
            for var_name, var in dataset.data_vars.items()
            if len(var.dims) >= 1
        ]
    else:
        var_names = []

    if var_names:
        if not var_name_1 or var_name_1 not in var_names:
            var_name_1 = var_names[0]
        if not var_name_2 or var_name_2 not in var_names:
            var_name_2 = var_names[0]

    return var_names, var_name_1, var_name_2
