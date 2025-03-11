import pandas as pd
import geopandas as gpd
import altair as alt
import pyproj
import shapely
import shapely.ops
from shapely.geometry import Point

from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, Typography, Select, VegaChart

from xcube.webapi.viewer.contrib import Panel, get_dataset
from xcube.server.api import Context
from xcube.constants import CRS_CRS84
from xcube.core.extract import get_cube_values_for_points
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

    dataset = get_dataset(ctx, dataset_id)

    if theme_mode == "light":
        theme_mode = "default"

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

    # interesting, when multiple=True
    variable_names = get_variables(dataset)
    select_variables = ""  # variable_names , when multiple=True
    select_variables = Select(
        id="select_variables",
        label="variables",
        value=select_variables,
        options=variable_names,
        # multiple=True,
    )

    button = Button(
        id="button", text="Update"  # "ADD Place to Spectral View"
    )  # , style={"maxWidth": 100})

    controls = Box(
        children=[select_places, select_variables, button],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": 6,
            "padding": 6,
            "flexGrow": 0,
        },
    )

    time = Component(id="time", children=[time_label])  # [None])

    return Box(
        children=[place_text, plot, controls, time],
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
    placegroup: gpd.GeoDataFrame,  # list[dict],
    places: list,
    # variables: list, when multiple=True
) -> pd.DataFrame:

    grid_mapping = GridMapping.from_dataset(dataset)

    # if place_geometry is not None and not grid_mapping.crs.is_geographic:
    project = pyproj.Transformer.from_crs(
        CRS_CRS84, grid_mapping.crs, always_xy=True
    ).transform

    placegroup["geometry"] = placegroup["geometry"].apply(
        lambda geom: shapely.ops.transform(project, geom)
    )
    placegroup["x"] = placegroup["geometry"].apply(
        lambda geom: geom.x if geom else None
    )
    placegroup["y"] = placegroup["geometry"].apply(
        lambda geom: geom.y if geom else None
    )

    dataset_place = get_cube_values_for_points(dataset, placegroup, include_refs=True)
    # dataset_place = get_cube_values_for_points(dataset, placegroup, include_refs=True, var_names=variables), when multiple =True

    result = pd.DataFrame()

    for place in places:

        i = (dataset_place.name_ref == place).argmax().item()
        selected_values = (
            dataset_place.drop_vars("geometry_ref")
            .sel(idx=i)
            .compute()
            .to_dict()["data_vars"]
        )

        variables = list(selected_values.keys())  # List of variable names
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
# TODO - this has to trigger a plot update if sth. changes
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
    State("@app", "selectedPlaceGroup"),
    State("select_places", "value"),
    State("select_variables", "value"),
    Input("button", "clicked"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    placegroup: list[str],
    place: list,
    variables: list,
    _clicked: bool | None = None,
) -> alt.Chart | None:

    if placegroup is None:
        return None

    if place is None:
        return None

    dataset = get_dataset(ctx, dataset_id)

    placegroup = gpd.GeoDataFrame(
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
            for feature in placegroup[0]["features"]
        ]
    )

    placegroup["time"] = pd.to_datetime(time_label).tz_localize(None)
    place = [place]  # update, becomes obsolete with multiple select
    source = get_wavelength(dataset, placegroup, place)
    # source = get_wavelength(dataset, placegroup, place,variables) # with multiple selection

    if source is None:
        # TODO: set error message in panel UI
        print("No reflectances found in Variables")
        return None

    chart = (
        alt.Chart(source).mark_line(point=True)
        # .mark_line()
        .encode(
            x="wavelength:Q",
            y="reflectance:Q",
            color="places:N",
            tooltip=["variable", "wavelength", "reflectance"],
        )
    ).properties(width=560, height=260)

    return chart


def get_places(place_group: list[dict]) -> list[str]:
    return [feature["properties"]["label"] for feature in place_group[0]["features"]]


def get_variables(dataset) -> list[str]:

    variables = [var for var in dataset.data_vars if "wavelength" in dataset[var].attrs]
    return variables


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
@panel.callback(
    State("@app", "selectedDatasetId"),
    State("@app", "selectedTimeLabel"),
    State("@app", "selectedPlaceGroup"),
    State("select_places", "value"),
    State("select_variables", "value"),
    State("time", "children"),
    Input("@app", "selectedTimeLabel"),
    #  Output("text", "children"),
    Output("plot", "chart"),
)
def update_timestep(
    ctx: Context,
    dataset_id: str,
    time_label: float,
    placegroup: str,
    place: list,
    variables: str,
    time: list,
    _new_time_label: bool | None = None,  # trigger, will always be True
) -> alt.Chart | None:  # tuple[list, alt.Chart] | None:

    # text = f"{dataset_id} " f"/ {time_label[0:-1]}"

    if time[0] != time_label:
        chart = update_plot(ctx, dataset_id, time_label, placegroup, place, variables)
        #  update_last_timelabel(time_label)
        return chart


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


# TODO implement this more efficient
@panel.callback(
    Output("time", "children"),
)
def update_last_timelabel(
    new_timelabel: str,
) -> list[str]:

    return [new_timelabel]
