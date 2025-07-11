#  Copyright (c) 2018-2025 by xcube team and contributors
#  Permissions are hereby granted under the terms of the MIT License:
#  https://opensource.org/licenses/MIT.
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import pyproj
import shapely
import shapely.geometry
import shapely.ops
from chartlets import Component, Input, Output, State
from chartlets.components import (
    Box,
    Button,
    CircularProgress,
    Select,
    VegaChart,
    Typography,
)

from xcube.constants import CRS_CRS84
from xcube.core.geom import mask_dataset_by_geometry, normalize_geometry
from xcube.core.gridmapping import GridMapping
from xcube.server.api import Context
from xcube.webapi.viewer.components import Markdown
from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.webapi.viewer.contrib.helpers import get_place_label

panel = Panel(__name__, title="2D Histogram (Demo)", icon="equalizer", position=4)


# Number of bins in x and y directions.
# This results in columns of 4096 items.
# Vega Altair's maximum is 5000.
NUM_BINS_MAX = 64


@panel.layout(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedDatasetTitle"),
    State("@app", "selectedTimeLabel"),
)
def render_panel(
    ctx: Context,
    dataset_id: str | None = None,
    dataset_title: str | None = None,
    time_label: str | None = None,
) -> Component:
    dataset = get_dataset(ctx, dataset_id)

    plot = VegaChart(
        id="plot",
        chart=None,
        style={
            "paddingTop": 6,
            # Since for dynamic resizing we use `container` as width and height for
            # this chart during updates, it is necessary that we provide the width
            # and the height here. This is for the `container` div of VegaChart.
            "width": "100%",
            "height": 400,
        },
    )

    if time_label:
        text = f"{dataset_title} / {time_label[0:-1]}"
    else:
        text = f"{dataset_title}"
    place_text = Typography(id="text", children=[text], align="left")

    var_names, var_name_1, var_name_2 = get_var_select_options(dataset)

    select_var_1 = Select(
        id="select_var_1", label="Variable 1", value=var_name_1, options=var_names
    )
    select_var_2 = Select(
        id="select_var_2", label="Variable 2", value=var_name_2, options=var_names
    )

    button = Button(id="button", text="Update", style={"maxWidth": 100}, disabled=True)

    controls = Box(
        children=[select_var_1, select_var_2, button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
        },
    )

    control_bar = Box(
        children=[place_text, controls],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "justifyContent": "space-between",
            "width": "100%",
            "gap": 6,
        },
    )

    error_message = Typography(
        id="error_message",
        style={"color": "red"},
        children=[""],
    )

    instructions = Typography(
        id="instructions",
        children=[
            Markdown(
                text=(
                    "Create or select a region shape in the map, then select "
                    "two variables from the dropdowns, and press **Update** "
                    "to create a 2D histogram plot."
                ),
            )
        ],
        variant="caption",
        color="textSecondary",
    )

    return Box(
        children=[
            instructions,
            control_bar,
            plot,
            error_message,
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "left",
            "width": "100%",
            "height": "100%",
            "gap": 6,
        },
    )


error_message = ""


@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedPlaceGeometry"),
    State("select_var_1"),
    State("select_var_2"),
    State("@app", "selectedTimeLabel"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    place_geometry: str | None = None,
    var_1_name: str | None = None,
    var_2_name: str | None = None,
    time_label: float | None = None,
    _clicked: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:
    global error_message
    dataset = get_dataset(ctx, dataset_id)

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

    if place_geometry is None or isinstance(place_geometry, shapely.geometry.Point):
        error_message = "Selected geometry must cover an area."
        return None

    dataset = mask_dataset_by_geometry(dataset, place_geometry)
    if dataset is None:
        error_message = "Selected geometry produces empty subset"
        return None

    var_1_data: np.ndarray = dataset[var_1_name].values.ravel()
    var_2_data: np.ndarray = dataset[var_2_name].values.ravel()
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
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x, y = np.meshgrid(x_centers, y_centers)
    # z = hist2d
    z = np.where(hist2d > 0.0, hist2d, np.nan).T
    source = pd.DataFrame(
        {var_1_name: x.ravel(), var_2_name: y.ravel(), "z": z.ravel()}
    )
    x_centers = x_edges[0:-1] + np.diff(x_edges) / 2
    y_centers = y_edges[0:-1] + np.diff(y_edges) / 2

    # Limit number of ticks on axes
    x_num_ticks = 8
    y_num_ticks = 8

    # Get the tick values using the center values
    x_tick_values = np.linspace(min(x_centers), max(x_centers), x_num_ticks)
    x_tick_values = np.array(
        [min(x_centers, key=lambda xc: abs(xc - t)) for t in x_tick_values]
    )

    y_tick_values = np.linspace(min(y_centers), max(y_centers), y_num_ticks)
    y_tick_values = np.array(
        [min(y_centers, key=lambda yc: abs(yc - t)) for t in y_tick_values]
    )

    chart = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X(
                f"{var_1_name}:O",
                # axis=alt.Axis(values=x_axis),
                axis=alt.Axis(
                    labelAngle=45,
                    values=x_tick_values,
                    labelOverlap="greedy",
                    labelPadding=5,
                    format=".3f",
                ),
                # scale=alt.Scale(bins=x_centers),
                scale=alt.Scale(nice=True),
            ),
            y=alt.Y(
                f"{var_2_name}:O",
                sort="descending",
                # scale=alt.Scale(bins=y_centers),
                # axis=alt.Axis(values=y_axis),
                scale=alt.Scale(nice=True),
                axis=alt.Axis(
                    values=y_tick_values,
                    labelOverlap="greedy",
                    labelPadding=5,
                    format=".3f",
                ),
            ),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="viridis"), title="Density"),
            tooltip=[var_1_name, var_2_name, "z:Q"],
        )
    ).properties(
        # allow chart to be adjusted to available container (<div>) size. Make sure
        # that you add width and height to the style props while defining the Vega
        # chart plot in render panel method
        width="container",
        height="container",
    )
    error_message = ""
    return chart


@panel.callback(
    Input("@app", "selectedDatasetId"),
    Input("@app", "selectedPlaceGeometry"),
    Output("button", "disabled"),
)
def set_button_disablement(
    _ctx: Context,
    dataset_id: str | None = None,
    place_geometry: str | None = None,
) -> bool:
    return not dataset_id or not place_geometry


@panel.callback(
    Input("@app", "selectedDatasetId"),
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


@panel.callback(
    State("@app", "selectedDatasetTitle"),
    State("@app", "selectedPlaceId"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "selectedTimeLabel"),
    Input("button", "clicked"),
    Output("text", "children"),
)
def update_text(
    ctx: Context,
    dataset_title: str,
    place_id: str | None = None,
    place_group: list[dict[str, Any]] | None = None,
    time_label: str | None = None,
    _clicked: bool | None = None,
) -> list | None:
    place_name = get_place_label(place_id, place_group)
    if time_label:
        return [f"{dataset_title} / {time_label[0:-1]} / {place_name}"]
    return [f"{dataset_title} "]


# TODO: Doesn't work. We need to ensure that show_progress() returns
#   before update_plot(). EDIT: This cannot work in its current form!
# @panel.callback(
#     Input("button", "clicked"),
#     Output("button", ""),
# )
def show_progress(
    _ctx: Context,
    _clicked: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:
    return CircularProgress(id="button", size=28)


@panel.callback(
    Input("@app", "selectedDatasetId"),
    Input("@app", "selectedPlaceGeometry"),
    Input("@app", "selectedTimeLabel"),
    State("select_var_1"),
    State("select_var_2"),
    Input("button", "clicked"),
    Output("error_message", "children"),
)
def update_error_message(
    ctx: Context,
    dataset_id: str | None = None,
    place_geometry: str | None = None,
    _time_label: str | None = None,
    var_1_name: str | None = None,
    var_2_name: str | None = None,
    _clicked: bool | None = None,
) -> str:
    global error_message

    if error_message == "":
        if dataset_id is None:
            error_message = "Missing dataset selection"

        if not place_geometry:
            error_message = "Missing place geometry selection"

        elif not var_1_name or not var_2_name:
            error_message = "Missing variable selection"

    return error_message
