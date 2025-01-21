import pandas as pd
from typing import Any, Union
import altair as alt
import pyproj
import shapely
import shapely.ops

from chartlets import Component, Input, State, Output, Container

##from chartlets.components import Box, Button, Typography, VegaChart
from chartlets.components import Box, Button, Typography, Plot  # VegaChart

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.webapi.viewer.contrib import get_datasets_ctx
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.geom import mask_dataset_by_geometry, normalize_geometry
from xcube.core.gridmapping import GridMapping

panel = Panel(__name__, title="Spectral View")


@panel.layout(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    #    State("@app", "selectedPlaceGeometry"),
    #    State("@app", "selectedPlaceId"),
    #    State("@app", "selectedVariableName"),
    State("@app", "themeMode"),
    #    State("@app", "selectedPlaceGroup"),
)
def render_panel(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    #    place_geometry: dict[str, Any],
    #    place_id: str,
    #    variable_name: str,
    theme_mode: str,
    #    placegroup: str,
) -> Component:

    if theme_mode == "dark":
        alt.theme.enable(name=theme_mode)
    else:
        alt.theme.enable(name="default")  # in viewer: light

    # dataset = get_dataset(ctx, dataset_id)

    # wavelengths = {}
    # for var_name, var in dataset.items():
    #     if "wavelength" in var.attrs:
    #         wavelengths[var_name] = var.attrs["wavelength"]

    plot = Plot(id="plot", chart=None, style={"flexGrow": 3})  # , theme="dark")

    text = (
        #  f"{ds_configs[0]['Title']} "
        f"{dataset_id} "
        f"/ {time_label[0:-1]}"  # / "
        # f"{round(place_geometry['coordinates'][0], 3)},"
        # f"{round(place_geometry['coordinates'][1], 3)} / "
        # f"{variable_name} /  "
        # f"{place_id}"
    )

    place_text = Typography(
        id="text", children=[text], color="pink", style={"flexGrow": 3}
    )

    button = Button(
        id="button", text="UPDATE Spectral View"
    )  # , style={"maxWidth": 100})

    controls = Box(
        children=[button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
            "padding": 6,
            "flexGrow": 0,
        },
    )

    reflectences = Container(id="reflectances", children=list())

    return Box(
        children=[place_text, plot, controls],
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
    dataset, time_label: float, place_geometry: dict[str, Any], place_id: str
) -> pd.DataFrame:

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

    # TODO: Error, find no gridmapping

    #  dataset = mask_dataset_by_geometry(dataset, place_geometry)
    if dataset is None:
        # TODO: set error message in panel UI
        print("dataset is None after masking, invalid geometry?")
        return None

    # TODO before that - use mask_by_geometry
    dataset = dataset.sel(
        y=place_geometry.y,
        x=place_geometry.x,
        method="nearest",
    )

    variables = []
    wavelengths = []
    for var_name, var in dataset.items():
        if "wavelength" in var.attrs:
            wavelengths.append(var.attrs["wavelength"])
            variables.append(var_name)

    result = []
    for var in variables:
        value = dataset[var].values.item()
        result.append({"place": place_id, "variable": var, "reflectance": value})

    result = pd.DataFrame(result)
    result["wavelength"] = wavelengths
    # print(result)
    # if not source.empty:
    source = pd.DataFrame()
    print("source")
    print(source)
    # if source is not None:
    if not source.empty:
        result = pd.concat([source, result])
    return result


# TODO - add selectedDatasetName to Available State Properties
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    Input("@app", "selectedDatasetId"),
    Input("@app", "selectedTimeLabel"),
    Output("text", "children"),
)
def update_text(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    _dataset: bool | None = None,
    _time_label: bool | None = None,
) -> list | None:

    # dataset = get_dataset(ctx, dataset_id)
    # title = dataset.attrs.get("title", "No title attribute found")
    # ds_ctx = get_datasets_ctx(ctx)
    # ds_configs = ds_ctx.get_dataset_configs()
    #
    # text = f"{ds_configs[0]['Title']} " f"/ {time_label[0:-1]}"
    # text = f"{title} " f"/ {time_label[0:-1]}"

    text = f"{dataset_id} " f"/ {time_label[0:-1]}"

    return [text]


@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGeometry"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "selectedPlaceId"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_geometry: dict[str, Any],
    placegroup: str,
    placeid: str,
    #   theme_mode: str,
    _clicked: bool | None = None,
    #   _new_theme: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:

    print("placeid")
    print(placeid is None)
    print("place geom")
    print(place_geometry)
    print("place group")
    print(placegroup)

    if placeid is None:
        # TODO: set error message in panel UI
        print("There is no place selected.")
        return None

    dataset = get_dataset(ctx, dataset_id)

    for feature in placegroup[0]["features"]:
        if feature["id"] == placeid:
            placelabel = feature["properties"]["label"]

    source = get_wavelength(dataset, time_label, place_geometry, placelabel)
    # source = pd.DataFrame(
    # {'wavelength': [1, 2, 3, 4], 'reflectance': [1, 2, 3, 4]}
    # )
    print(source)
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
            color="place:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(
        width=560, height=260
    )  # title="Spectral Chart",

    return chart


# TODO when a theme is changed, the new plot gets generated for the currently
# selected place, so the old plot gets lost.
# It is needed to find a way to save the place name of the plot to create a plot for
# the same place again.
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGeometry"),
    State("@app", "selectedPlaceGroup"),
    State("@app", "selectedPlaceId"),
    State("@app", "themeMode"),
    State("plot", "chart"),
    Input("@app", "themeMode"),
    Output("plot", "chart"),
)
def update_theme(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    place_geometry: dict[str, Any],
    placegroup: str,
    placeid: str,
    theme_mode: str,
    chart: alt.Chart,
    _new_theme: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:

    if theme_mode == "light":
        theme_mode = "default"

    if alt.theme.active != theme_mode:
        alt.theme.enable(name=theme_mode)

        chart = update_plot(
            ctx, dataset_id, time_label, place_geometry, placegroup, placeid
        )

    return chart


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
