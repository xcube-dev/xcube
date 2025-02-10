import pandas as pd
from typing import Any, Union
import altair as alt
import pyproj
import shapely
import shapely.ops

from chartlets import Component, Input, State, Output, Container

# from chartlets.components import Box, Button, Typography, Plot, Select  # VegaChart

from chartlets.components import Box, Button, Typography, Select, VegaChart

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.geom import mask_dataset_by_geometry, normalize_geometry
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectral View")


@panel.layout(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "themeMode"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_group: list[str],
    theme_mode: str,
) -> Component:

    # if theme_mode == "dark":
    #    alt.theme.enable(name=theme_mode)
    # else:
    #    alt.theme.enable(name="default")  # in viewer: light
    if theme_mode == "light":
        theme_mode = "default"
    # plot = Plot(id="plot", chart=None, style={"flexGrow": 3})  # , theme="dark")
    plot = VegaChart(id="plot", chart=None, style={"flexGrow": 3}, theme=theme_mode)

    text = f"{dataset_id} " f"/ {time_label[0:-1]}"

    place_text = Typography(
        id="text", children=[text], color="pink", style={"flexGrow": 3}
    )

    selected_places = ""
    place_names = get_places(place_group)
    select_places = Select(
        id="select_places",
        label="places",
        value=selected_places,
        options=place_names,
        # multiple=True,
    )

    button = Button(
        id="button", text="Update"  # "ADD Place to Spectral View"
    )  # , style={"maxWidth": 100})

    controls = Box(
        children=[select_places, button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
            "padding": 6,
            "flexGrow": 0,
        },
    )

    places = Component(id="places", children=[])  # [None])

    return Box(
        children=[place_text, plot, controls, places],
        style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "width": "100%",
            "height": "100%",
            #    "gap": 6,
            #    "padding": 6,
        },
    )


def get_wavelength(
    dataset,
    time_label: float,
    placegroup: list[str],
    place: list,
) -> pd.DataFrame:

    result = pd.DataFrame()

    if "time" in dataset.coords:
        if time_label:
            dataset = dataset.sel(time=pd.Timestamp(time_label[0:-1]), method="nearest")
        else:
            dataset = dataset.isel(time=-1)

    grid_mapping = GridMapping.from_dataset(dataset)

    for feature in placegroup[0]["features"]:
        print("PLACE!!!!!!!!!!!!!!!!!!!")
        print(place)
        print(feature["id"])
        # if feature["id"] == place:
        if feature["properties"]["label"] == place:
            print("TRUEEEEEEEEEEEEEEEE")
            placelabel = feature["properties"]["label"]
            place_geometry = feature["geometry"]

            print("placelabel")
            print(placelabel)
            print("placegeom")
            print(place_geometry)
            place_geometry = normalize_geometry(place_geometry)
            if place_geometry is not None and not grid_mapping.crs.is_geographic:

                project = pyproj.Transformer.from_crs(
                    CRS_CRS84, grid_mapping.crs, always_xy=True
                ).transform
                place_geometry = shapely.ops.transform(project, place_geometry)

            print(place_geometry)
            # TODO: Error, find no gridmapping

            #  dataset = mask_dataset_by_geometry(dataset, place_geometry)
            if dataset is None:
                # TODO: set error message in panel UI
                print("dataset is None after masking, invalid geometry?")
                return None

            print(place_geometry.y)
            print(place_geometry.x)
            # TODO before that - use mask_by_geometry or get value_for_point
            dataset_place = dataset.sel(
                y=place_geometry.y,
                x=place_geometry.x,
                method="nearest",
            )

            print("get wavelength")
            variables = []
            wavelengths = []
            for var_name, var in dataset_place.items():
                if "wavelength" in var.attrs:
                    wavelengths.append(var.attrs["wavelength"])
                    variables.append(var_name)

            res = []
            for var in variables:
                # print(var)
                value = dataset_place[var].values.item()
                res.append(
                    {"places": placelabel, "variable": var, "reflectance": value}
                )

            res = pd.DataFrame(res)
            res["wavelength"] = wavelengths

            result = pd.concat([result, res])

    print(result)
    return result


# TODO - add selectedDatasetName to Available State Properties
# TODO - this has to trigger an updated plot if sth. changes
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    Input("@app", "selectedTimeLabel"),
    Output("text", "children"),
)
def update_text(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    _time_label: bool | None = None,
) -> list | None:

    text = f"{dataset_id} " f"/ {time_label[0:-1]}"

    return [text]


@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGeometry"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "selectedPlaceId"),
    State("select_places", "value"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_geometry: dict[str, Any],
    placegroup: list[str],
    placeid: str,
    place: list,
    _clicked: bool | None = None,
) -> alt.Chart | None:

    print("placeid")
    print(placeid is None)
    print("place geom")
    print(place_geometry)
    print("place group")
    print(placegroup)
    print("places")
    print(place)

    if placegroup is None:
        return None

    if place is None:
        return None

    dataset = get_dataset(ctx, dataset_id)

    source = get_wavelength(dataset, time_label, placegroup, place)

    if source is None:
        # TODO: set error message in panel UI
        print("No reflectances found in Variables")
        return None

    chart = (
        alt.Chart(source)
        # .mark_line(point=True)
        .mark_line().encode(
            x="wavelength:Q",
            y="reflectance:Q",
            color="places:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(
        width=560, height=260
    )  # title="Spectral Chart",

    return chart


@panel.callback(
    State("@app", "themeMode"), Input("@app", "themeMode"), Output("plot", "theme")
)
def update_theme(
    ctx: Context,
    theme_mode: str,
    _new_theme: bool | None = None,
) -> str:

    if theme_mode == "light":
        theme_mode = "default"

    return theme_mode


# # TODO - add selectedDatasetName to Available State Properties
# @panel.callback(
#     State("@app", "selectedDatasetId"),
#     State("@app", "selectedTimeLabel"),
#     State("@app", "selectedPlaceGeometry"),
#     State("@app", "selectedPlaceGroup"),
#     State("@app", "selectedPlaceId"),
#     Input("@app", "selectedTimeLabel"),
#     #  Output("text", "children"),
#     Output("plot", "chart"),
# )
# def update_timestep(
#     ctx: Context,
#     dataset_id: str,
#     time_label: float,
#     place_geometry: dict[str, Any],
#     placegroup: str,
#     placeid: str,
#     _new_time_label: bool | None = None,  # trigger, will always be True
# ) -> alt.Chart | None:  # tuple[list, alt.Chart] | None:
#
#     text = f"{dataset_id} " f"/ {time_label[0:-1]}"
#
#     chart = update_plot(
#         ctx, dataset_id, time_label, place_geometry, placegroup, placeid
#     )
#     return chart
#
#     # return [text], chart


@panel.callback(
    State("@app", "selectedPlaceId"),
    State("places", "children"),
    # Input("button", "clicked"),
    Output("places", "children"),
)
def add_place(
    ctx: Context,
    placeid: str,
    places: list,
    #  _clicked: bool | None = None,
) -> list:

    print(places)
    if placeid is None:
        # TODO: set error message in panel UI
        print("There is no place selected.")
        return None

    if places is None:
        places = [placeid]
    else:
        if placeid not in places:
            places.append(placeid)

    print(places)
    return places


# @panel.callback(
#     State("places", "children"),
#     Input("button_reset", "clicked"),
#     Output("places", "children"),
# )
# def clear_places(
#     ctx: Context,
#     places: list,
#     _clicked: bool | None = None,
# ) -> list:
#
#     print(places)
#     if places is None:
#         print("nothing to reset")
#     else:
#         places = []
#
#     print(places)
#     return places


def get_places(place_group: list[dict]) -> list[str]:
    return [feature["properties"]["label"] for feature in place_group[0]["features"]]


@panel.callback(
    State("@app", "selectedPlaceGroup"),
    Input("@app", "selectedPlaceGroup"),
    Output("select_places", "options"),
)
def update_places(
    ctx: Context,
    place_group: str,
    _new: bool | None = None,
) -> list[str]:

    return [feature["properties"]["label"] for feature in place_group[0]["features"]]
