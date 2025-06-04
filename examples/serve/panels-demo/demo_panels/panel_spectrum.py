from typing import Any

import altair as alt
import geopandas as gpd
import pandas as pd
import pyproj
import shapely
import shapely.ops
from shapely.geometry import Point
import xarray as xr

from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, Typography, Select, VegaChart

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.extract import get_cube_values_for_points
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectrum View (Demo)", icon="light", position=4)


@panel.layout(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "themeMode"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: str,
    place_group: list[dict[str, Any]],
    theme_mode: str,
) -> Component:

    if theme_mode == "light":
        theme_mode = "default"

    plot = VegaChart(
        id="plot",
        chart=None,
        style={"paddingTop": 6, "width": "100%", "height": 400},
        theme=theme_mode,
    )
    if time_label:
        text = f"{dataset_id} / {time_label[0:-1]}"
    else:
        text = f"{dataset_id}"
    place_text = Typography(id="text", children=[text], align="left")

    # Ideas
    # 1. Adding radio-button for two modes:
    #    update mode - reactive to changes to dataset/places/time/variables
    #    active mode -

    # How should spectrum viewer behave?
    # It should be reactive to changes to dataset/places/times
    # How to freeze the current spectrum?
    # We add a button to add it permanently to the graph and then when a new point is
    # selected it becomes reactive again.

    # Second mode - Just change for time but new line in graph for a new point

    # First version: Reactive to time and place changes
    # Add button: Would add the  spectrum view of current time and place to the graph
    # with legend place/time (static)
    # Delete button: Would delete the last one the spectrum views in the plot
    # Move the text align to left

    # Make line chart and bar chart

    add_button = Button(id="add_button", text="Add", style={"maxWidth": 100})

    controls = Box(
        children=[add_button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
            "padding": 6,
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
        id="error_message", style={"color": "red"}, children=[""]
    )

    return Box(
        children=[
            "Select a map point from the dropdown and press 'Update' "
            "to create a spectrum plot for that point and the selected time.",
            control_bar,
            error_message,
            plot,
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "width": "100%",
            "height": "100%",
            "gap": 6,
        },
    )


def get_spectra(
    dataset: xr.Dataset,
    place_group: gpd.GeoDataFrame,
    places: list,
) -> pd.DataFrame:

    grid_mapping = GridMapping.from_dataset(dataset)

    # if place_geometry is not None and not grid_mapping.crs.is_geographic:
    project = pyproj.Transformer.from_crs(
        CRS_CRS84, grid_mapping.crs, always_xy=True
    ).transform

    place_group["geometry"] = place_group["geometry"].apply(
        lambda geom: shapely.ops.transform(project, geom)
    )
    place_group["x"] = place_group["geometry"].apply(
        lambda geom: geom.x if geom else None
    )
    place_group["y"] = place_group["geometry"].apply(
        lambda geom: geom.y if geom else None
    )

    dataset_place = get_cube_values_for_points(dataset, place_group, include_refs=True)

    result = pd.DataFrame()

    for place in places:

        i = (dataset_place.name_ref == place).argmax().item()
        selected_values = (
            dataset_place.drop_vars("geometry_ref")
            .sel(idx=i)
            .compute()
            .to_dict()["data_vars"]
        )

        variables = list(selected_values.keys())
        values = [selected_values[var]["data"] for var in variables]
        wavelengths = [
            dataset_place[var].attrs.get("wavelength", None) for var in variables
        ]

        res = {
            "places": place,
            "variable": variables,
            "reflectance": values,
            "wavelength": wavelengths,
        }

        res = pd.DataFrame(res)
        result = pd.concat([result, res])

    result = result.dropna(subset=["wavelength"])
    return result


@panel.callback(
    State("@app", "selectedDatasetTitle"),
    Input("@app", "selectedTimeLabel"),
    Output("text", "children"),
)
def update_text(
    ctx: Context,
    dataset_title: str | None = None,
    time_label: str | None = None,
) -> list | None:

    if time_label:
        return [f"{dataset_title} " f"/ {time_label[0:-1]}"]
    return [f"{dataset_title} "]


@panel.callback(
    State("@app", "selectedDatasetId"),
    Input("@app", "selectedTimeLabel"),
    Input("@app", "selectedPlaceGeometry"),
    State("@app", "selectedPlaceGroup"),
    Input("add_button", "clicked"),
    Input("plot", "chart"),
    Output("plot", "chart"),
    Output("error_message", "children"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    time_label: str | None = None,
    place_geo: dict[str, Any] | None = None,
    place_group: list[dict[str, Any]] | None = None,
    _clicked: bool | None = None,
    chart=None,
) -> tuple[alt.Chart | None, str]:
    print("clicked", _clicked)
    print("chart", chart)
    dataset = get_dataset(ctx, dataset_id)
    has_point = any(
        feature.get("geometry", {}).get("type") == "Point"
        for collection in place_group
        for feature in collection.get("features", [])
    )
    if dataset is None:
        return None, "Missing dataset selection"
    elif not place_group or not has_point:
        return None, "Missing point selection"

    label = find_selected_point_label(place_group, place_geo)

    if label is None:
        return None, "There is no label for the selected point"

    if place_geo.get("type") == "Point":
        place_group = gpd.GeoDataFrame(
            [
                {
                    "name": label,
                    "x": place_geo["coordinates"][0],
                    "y": place_geo["coordinates"][1],
                    "geometry": Point(
                        place_geo["coordinates"][0],
                        place_geo["coordinates"][1],
                    ),
                }
            ]
        )
    else:
        return None, "Selected geometry must be a point"

    place_group["time"] = pd.to_datetime(time_label).tz_localize(None)
    places_select = [label]
    source = get_spectra(dataset, place_group, places_select)

    if source is None:
        return None, "No reflectances found in Variables"

    chart = (
        alt.Chart(source)
        .mark_bar(point=True)
        .encode(
            x="wavelength:Q",
            y="reflectance:Q",
            color="places:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(width="container", height="container")

    return chart, ""


@panel.callback(
    Input("@app", "selectedPlaceGeometry"),
    Output("add_button", "disabled"),
)
def set_button_disablement(
    _ctx: Context,
    place_geometry: str | None = None,
) -> bool:
    print("in set_button_disablement", place_geometry)
    return not place_geometry


def find_selected_point_label(features_data, target_point):
    for feature_collection in features_data:
        for feature in feature_collection.get("features", []):
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            geo_type = geometry.get("type", "")

            if (
                coordinates == target_point["coordinates"]
                and geo_type == target_point["type"]
            ):
                return feature.get("properties", {}).get("label", None)

    return None
