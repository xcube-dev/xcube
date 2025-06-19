import math
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
    Typography,
    VegaChart,
    Radio,
    RadioGroup,
)

from xcube.webapi.viewer.components import Markdown
from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.extract import get_cube_values_for_points
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectrum View (Demo)", icon="sunny", position=5)

_THROTTLE_TOTAL_SPECTRUM_PLOTS = 10


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

    update_radio = Radio(id="update_radio", value="update", label="Update")
    add_radio = Radio(id="add_radio", value="add", label="Add")

    exploration_radio_group = RadioGroup(
        id="exploration_radio_group",
        children=[add_radio, update_radio],
        label="Exploration Mode",
        style={
            "display": "flex",
            "flexDirection": "row",
        },
        tooltip=(
            "Add: Current spectrum is added and new point selections will be "
            "added as new spectra. Update: Clear the chart but the current "
            "selection if any. "
        ),
    )

    control_bar = Box(
        children=[place_text, exploration_radio_group],
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

    instructions = Typography(
        id="instructions",
        children=[
            "Choose an exploration mode and select points to visualize "
            "their spectral reflectance across available wavelengths in "
            "this highly dynamic Spectrum View.",
            Markdown(
                text=(
                    "_Note: Only 10 spectra can be added at a time as older "
                    "ones are removed. When switching from **Add** to **Update** "
                    "mode, the existing bar plots will be cleared if any._"
                )
            ),
        ],
        variant="caption",
        color="textSecondary",
    )

    return Box(
        children=[
            instructions,
            control_bar,
            error_message,
            plot,
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
    State("@container", "spectrum_list"),
    State("@container", "previous_mode"),
    Input("exploration_radio_group", "value"),
    State("plot", "chart"),
    Output("plot", "chart"),
    Output("error_message", "children"),
    Output("@container", "spectrum_list"),
    Output("@container", "previous_mode"),
)
def update_plot(
    ctx: Context,
    dataset_id: str | None = None,
    time_label: str | None = None,
    place_geo: dict[str, Any] | None = None,
    place_group: list[dict[str, Any]] | None = None,
    spectrum_list: list[str] | None = None,
    previous_mode: str | None = None,
    exploration_radio_group: str | None = None,
    current_chart: alt.Chart | None = None,
) -> tuple[alt.Chart | None, str, list, str]:
    if exploration_radio_group is None:
        return None, "Missing exploration mode choice", spectrum_list, previous_mode

    dataset = get_dataset(ctx, dataset_id)
    has_point = any(
        feature.get("geometry", {}).get("type") == "Point"
        for collection in place_group
        for feature in collection.get("features", [])
    )

    if dataset is None:
        return None, "Missing dataset selection", spectrum_list, exploration_radio_group
    elif not place_group or not has_point:
        return None, "Missing point selection", spectrum_list, exploration_radio_group

    label = find_selected_point_label(place_group, place_geo)

    if label is None:
        return (
            None,
            "There is no label for the selected point or no point is selected",
            spectrum_list,
            previous_mode,
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
        return None, "Selected geometry must be a point", spectrum_list, previous_mode

    place_group_geodf["time"] = pd.to_datetime(time_label).tz_localize(None)
    places_select = [label]
    new_spectrum_data = get_spectra(dataset, place_group_geodf, places_select)

    if new_spectrum_data is None or new_spectrum_data.empty:
        return None, "No reflectances found in Variables", spectrum_list, previous_mode

    new_spectrum_data["Legend"] = new_spectrum_data["places"] + ": " + time_label

    existing_data = extract_data_from_chart(current_chart)

    # Filter points in case the user deletes them.
    valid_labels = {
        feature["properties"]["label"]
        for item in place_group
        for feature in item.get("features", [])
    }
    existing_data = filter_data_by_valid_labels(existing_data, valid_labels)

    if exploration_radio_group == "update":
        if previous_mode == "add":
            existing_data = pd.DataFrame()
        else:
            existing_data, spectrum_list = remove_last_added_place(
                existing_data, spectrum_list
            )

        updated_data = add_place_data_to_existing(existing_data, new_spectrum_data)
        if spectrum_list is None:
            spectrum_list = []
        spectrum_list.append(label)
    else:
        updated_data = add_place_data_to_existing(existing_data, new_spectrum_data)

    # Vega Altair doesn’t support xOffset with x:Q, so we manually shift each bar
    # slightly
    unique_groups = sorted(updated_data["Legend"].unique())
    n_groups = len(unique_groups)
    group_offset_map = {
        group: i - (n_groups - 1) / 2 for i, group in enumerate(unique_groups)
    }

    bar_spacing = 3
    updated_data["x_offset"] = updated_data.apply(
        lambda row: row["wavelength"] + group_offset_map[row["Legend"]] * bar_spacing,
        axis=1,
    )

    new_chart = create_chart_from_data(updated_data)
    return new_chart, "", spectrum_list, exploration_radio_group


def find_selected_point_label(
    features_data: list[dict[str, Any]], target_point: dict[str, Any]
) -> str | None:
    if target_point is None:
        return None
    for feature_collection in features_data:
        for feature in feature_collection.get("features", []):
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            geo_type = geometry.get("type", "")

            if coordinates == target_point.get(
                "coordinates", []
            ) and geo_type == target_point.get("type", ""):
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
                color="Legend:N",
                tooltip=["places", "variable", "wavelength", "reflectance"],
            )
            .configure_legend(orient="bottom", columns=2)
            .properties(width="container", height="container")
        )

    return (
        alt.Chart(data)
        .mark_bar(size=2)
        .encode(
            x=alt.X("x_offset:Q", title="Wavelength"),
            y=alt.Y("reflectance:Q", title="Reflectance"),
            xOffset="Legend:N",
            color="Legend:N",
            tooltip=["places", "variable", "wavelength", "reflectance"],
        )
        .configure_legend(orient="bottom", columns=2)
        .properties(width="container", height="container")
    )


def add_place_data_to_existing(
    existing_data: pd.DataFrame, new_data: pd.DataFrame
) -> pd.DataFrame:
    if new_data.empty:
        return existing_data

    if existing_data.empty:
        return new_data

    # This is to check if the new_data already exists in the existing_data to avoid
    # duplication
    if not existing_data.empty:
        merged = new_data.merge(existing_data, how="left", indicator=True)
        if (merged["_merge"] == "both").all():
            return existing_data

    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Throttling to last 10 spectrum views
    final_df = combined_data[
        combined_data["places"].isin(
            combined_data.drop_duplicates("places", keep="last").tail(
                _THROTTLE_TOTAL_SPECTRUM_PLOTS
            )["places"]
        )
    ]
    return final_df


def remove_last_added_place(
    data: pd.DataFrame, spectrum_list: list
) -> tuple[pd.DataFrame, list]:
    if not spectrum_list or data.empty:
        return data, spectrum_list

    last_places = spectrum_list.pop()
    filtered_data = data[~data["places"].isin([last_places])]
    return filtered_data, spectrum_list


def filter_data_by_valid_labels(data: pd.DataFrame, valid_labels: set) -> pd.DataFrame:
    if data.empty:
        return data
    return data[data["places"].isin(valid_labels)]
