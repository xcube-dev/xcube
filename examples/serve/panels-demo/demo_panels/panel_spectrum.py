import math
import json
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
from chartlets.components import (
    Box,
    Button,
    Typography,
    VegaChart,
    Radio,
    RadioGroup,
)

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.extract import get_cube_values_for_points
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectrum View (Demo)", icon="light", position=4)


@panel.layout(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "themeMode"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: str,
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

    active_radio = Radio(id="active_radio", value="active", label="Active Mode")
    save_radio = Radio(id="save_radio", value="save", label="Save Mode")

    exploration_radio_group = RadioGroup(
        id="exploration_radio_group",
        children=[active_radio, save_radio],
        label="Exploration Mode",
    )

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

    delete_button = Button(
        id="delete_button", text="Delete last point", style={"maxWidth": 100}
    )

    controls = Box(
        children=[exploration_radio_group, delete_button],
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

    places_stack_storage = Typography(
        id="places_stack_storage",
        children=["[]"],
        style={"display": "none"},
    )

    return Box(
        children=[
            "Choose an exploration mode and create/select points to view the Spectrum data.",
            control_bar,
            error_message,
            plot,
            places_stack_storage,
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
        cleaned_values = [
            0 if isinstance(x, float) and math.isnan(x) else x for x in values
        ]

        wavelengths = [
            dataset_place[var].attrs.get("wavelength", None) for var in variables
        ]

        res = {
            "places": place,
            "variable": variables,
            "reflectance": cleaned_values,
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
        return [f"{dataset_title} / {time_label[0:-1]}"]
    return [f"{dataset_title} "]


@panel.callback(
    State("@app", "selectedDatasetId"),
    Input("@app", "selectedTimeLabel"),
    Input("@app", "selectedPlaceGeometry"),
    State("@app", "selectedPlaceGroup"),
    State("exploration_radio_group", "value"),
    State("plot", "chart"),
    State("places_stack_storage", "children"),
    Output("plot", "chart"),
    Output("error_message", "children"),
    Output("places_stack_storage", "children"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    time_label: str | None = None,
    place_geo: dict[str, Any] | None = None,
    place_group: list[dict[str, Any]] | None = None,
    exploration_radio_group: str | None = None,
    current_chart: alt.Chart | None = None,
    places_stack_json: list | None = None,
) -> tuple[alt.Chart | None, str, list]:
    import json

    places_stack = []
    if places_stack_json and len(places_stack_json) > 0:
        try:
            places_stack = json.loads(places_stack_json[0])
        except (json.JSONDecodeError, IndexError):
            places_stack = []

    if exploration_radio_group is None:
        return None, "Missing exploration mode choice", [json.dumps(places_stack)]

    dataset = get_dataset(ctx, dataset_id)
    has_point = any(
        feature.get("geometry", {}).get("type") == "Point"
        for collection in place_group
        for feature in collection.get("features", [])
    )

    if dataset is None:
        return None, "Missing dataset selection", [json.dumps(places_stack)]
    elif not place_group or not has_point:
        return None, "Missing point selection", [json.dumps(places_stack)]

    label = find_selected_point_label(place_group, place_geo)

    if label is None:
        return (
            None,
            "There is no label for the selected point",
            [json.dumps(places_stack)],
        )

    if place_geo.get("type") == "Point":
        place_group_geodf = gpd.GeoDataFrame(
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
        return None, "Selected geometry must be a point", [json.dumps(places_stack)]

    place_group_geodf["time"] = pd.to_datetime(time_label).tz_localize(None)
    places_select = [label]
    new_spectrum_data = get_spectra(dataset, place_group_geodf, places_select)

    if new_spectrum_data is None or new_spectrum_data.empty:
        return None, "No reflectances found in Variables", [json.dumps(places_stack)]

    existing_data = extract_data_from_chart(current_chart)

    # Filter points in case the user deletes them.
    valid_labels = {
        feature["properties"]["label"]
        for item in place_group
        for feature in item.get("features", [])
    }
    existing_data = filter_data_by_valid_labels(existing_data, valid_labels)

    if exploration_radio_group == "active":
        if places_stack:
            existing_data, places_stack = remove_last_added_place(
                existing_data, places_stack
            )

        updated_data = add_place_data_to_existing(existing_data, new_spectrum_data)
        places_stack.append([label])
    else:
        updated_data = add_place_data_to_existing(existing_data, new_spectrum_data)
        places_stack.append([label])

    new_chart = create_chart_from_data(updated_data)

    return new_chart, "", [json.dumps(places_stack)]


@panel.callback(
    Input("delete_button", "clicked"),
    State("plot", "chart"),
    State("places_stack_storage", "children"),
    Output("plot", "chart"),
    Output("places_stack_storage", "children"),
)
def delete_places(
    ctx: Context,
    _clicked: bool | None = None,
    current_chart: alt.Chart | None = None,
    places_stack_json: list | None = None,
) -> tuple[alt.Chart, list]:
    places_stack = []
    if places_stack_json and len(places_stack_json) > 0:
        try:
            places_stack = json.loads(places_stack_json[0])
        except (json.JSONDecodeError, IndexError):
            places_stack = []

    current_data = extract_data_from_chart(current_chart)
    updated_data, updated_stack = remove_last_added_place(current_data, places_stack)
    new_chart = create_chart_from_data(updated_data)
    return new_chart, [json.dumps(updated_stack)]


@panel.callback(
    Input("@app", "selectedPlaceGeometry"),
    Input("exploration_radio_group", "value"),
    Output("delete_button", "disabled"),
)
def set_button_disablement(
    _ctx: Context,
    place_geometry: str | None = None,
    exploration_radio_group: str | None = None,
) -> bool:
    return not place_geometry and exploration_radio_group != "save"


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


def extract_data_from_chart(chart: alt.Chart) -> pd.DataFrame:
    if chart is None:
        return pd.DataFrame(columns=["places", "variable", "reflectance", "wavelength"])

    if chart.get("datasets", {}) != {}:
        return pd.DataFrame(list(chart.get("datasets").values())[0])

    return pd.DataFrame(columns=["places", "variable", "reflectance", "wavelength"])


def create_chart_from_data(data: pd.DataFrame) -> alt.Chart:
    if data.empty:
        return (
            alt.Chart(
                pd.DataFrame(
                    columns=["places", "variable", "reflectance", "wavelength"]
                )
            )
            .mark_bar()
            .encode(
                x="wavelength:N",
                y="reflectance:Q",
                xOffset="places:N",
                color="places:N",
                tooltip=["variable", "wavelength", "reflectance"],
            )
            .properties(width="container", height="container")
        )

    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x="wavelength:N",
            y="reflectance:Q",
            xOffset="places:N",
            color="places:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(width="container", height="container")


def add_place_data_to_existing(
    existing_data: pd.DataFrame, new_data: pd.DataFrame
) -> pd.DataFrame:
    if new_data.empty:
        return existing_data

    # This is to check if the new_data already exists in the existing_data to avoid
    # duplication
    if not existing_data.empty:
        merged = new_data.merge(existing_data, how="left", indicator=True)
        if (merged["_merge"] == "both").all():
            return existing_data

    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    return combined_data


def remove_last_added_place(
    data: pd.DataFrame, places_stack: list
) -> tuple[pd.DataFrame, list]:
    if not places_stack or data.empty:
        return data, places_stack

    last_places = places_stack.pop()
    filtered_data = data[~data["places"].isin(last_places)]
    return filtered_data, places_stack


def filter_data_by_valid_labels(data: pd.DataFrame, valid_labels: set) -> pd.DataFrame:
    if data.empty:
        return data
    return data[data["places"].isin(valid_labels)]
