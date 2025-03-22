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

    plot = VegaChart(id="plot", chart=None, style={"paddingTop": 6}, theme=theme_mode)

    text = f"{dataset_id} " f"/ {time_label[0:-1]}"
    place_text = Typography(id="text", children=[text], align="center")

    place_names = get_places(ctx, place_group)
    select_places = Select(
        id="select_places",
        label="places (points)",
        value="",
        options=place_names,
    )

    button = Button(id="button", text="Update", style={"maxWidth": 100})

    controls = Box(
        children=[select_places, button],
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

    return Box(
        children=[
            "Select a map point from the dropdown and press 'Update' "
            "to create a spectrum plot for that point and the selected time.",
            control_bar,
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


def get_wavelength(
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


# TODO - add selectedDatasetName to Available State Properties
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    Input("@app", "selectedTimeLabel"),
    Output("text", "children"),
)
def update_text(
    ctx: Context,
    dataset_id: str,
    time_label: str,
    _time_label: bool | None = None,
) -> list | None:

    text = f"{dataset_id} " f"/ {time_label[0:-1]}"

    return [text]


@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGroup"),
    State("select_places", "value"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str,
    time_label: str,
    place_group: list[dict[str, Any]],
    place: list,
    _clicked: bool | None = None,
) -> alt.Chart | None:

    if not place_group:
        return None

    if not place:
        return None

    dataset = get_dataset(ctx, dataset_id)

    place_group = gpd.GeoDataFrame(
        [
            {
                "id": feature["id"],
                "name": feature["properties"]["label"],
                "color": feature["properties"]["color"],
                "x": feature["geometry"]["coordinates"][0],
                "y": feature["geometry"]["coordinates"][1],
                "geometry": Point(
                    feature["geometry"]["coordinates"][0],
                    feature["geometry"]["coordinates"][1],
                ),
            }
            for feature in place_group[0]["features"]
            if feature.get("geometry", {}).get("type") == "Point"
        ]
    )

    place_group["time"] = pd.to_datetime(time_label).tz_localize(None)
    place = [place]
    source = get_wavelength(dataset, place_group, place)

    if source is None:
        # TODO: set error message in panel UI
        print("No reflectances found in Variables")
        return None

    chart = (
        alt.Chart(source)
        .mark_line(point=True)
        .encode(
            x="wavelength:Q",
            y="reflectance:Q",
            color="places:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(width=300, height=200)

    return chart


@panel.callback(
    Input("@app", "selectedPlaceGroup"),
    Output("select_places", "options"),
)
def get_places(
    ctx: Context,
    place_group: list[dict[str, Any]],
) -> list[str]:

    if not place_group:
        return []
    else:
        return [
            feature["properties"]["label"]
            for feature in place_group[0]["features"]
            if feature.get("geometry", {}).get("type") == "Point"
        ]


@panel.callback(
    State("@app", "themeMode"),
    Input("@app", "themeMode"),
    Output("plot", "theme"),
)
def update_theme(
    ctx: Context,
    theme_mode: str,
    _new_theme: bool | None = None,
) -> str:

    if theme_mode == "light":
        theme_mode = "default"

    return theme_mode


# TODO - add selectedDatasetName to Available State Properties
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGroup"),
    State("select_places", "value"),
    Input("@app", "selectedTimeLabel"),
    Output("plot", "chart"),
)
def update_timestep(
    ctx: Context,
    dataset_id: str,
    time_label: str,
    place_group: list[dict[str, Any]],
    place: list,
    _new_time_label: bool | None = None,
) -> alt.Chart | None:

    return update_plot(ctx, dataset_id, time_label, place_group, place)
