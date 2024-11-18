import altair
from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, Plot, Typography

from xcube.webapi.viewer.contrib import Panel
from xcube.webapi.viewer.contrib import get_dataset
from xcube.server.api import Context


panel = Panel(__name__, title="2D Histogram")


@panel.layout()
def render_panel(ctx: Context) -> Component:

    info = Typography(
        text="This plot requires a pinned variable" " and a place to be selected."
    )

    plot = Plot(
        id="plot",
        chart=None,
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",
            "gap": "6px",
        },
    )

    button = Button(id="button", text="Update", disabled=True, style={"maxWidth": 100})

    return Box(
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",
            "gap": 6,
            "padding": 6,
        },
        children=[info, plot, button],
    )


# noinspection PyUnusedLocal
@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    Input(source="app", property="controlState.selectedVariableName"),
    Input(source="app", property="controlState.selectedDataset2Id"),
    Input(source="app", property="controlState.selectedVariable2Name"),
    Input(source="app", property="controlState.selectedTimeLabel"),
    Input(source="app", property="controlState.selectedPlaceGeometry"),
    Input("button", "n_clicks"),
    Output("plot", "chart"),
)
def update_plot(
    ctx: Context,
    dataset_id_1: str = "",
    variable_name_1: str = "",
    dataset_id_2: str = 0,
    variable_name_2: str = "",
    time_label: float = 0,
    place_geometry: str = "",
    n_clicks: int | None = None,  # trigger
) -> altair.Chart | None:
    enabled = all(
        (
            dataset_id_1,
            dataset_id_2,
            variable_name_1,
            variable_name_2,
            time_label,
            place_geometry,
        )
    )
    if not enabled:
        return None
    dataset_1 = get_dataset(ctx, dataset_id_1)
    dataset_2 = get_dataset(ctx, dataset_id_2)
    variable_1 = dataset_1[variable_name_1]
    variable_2 = dataset_2[variable_name_2]


@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    Input(source="app", property="controlState.selectedVariableName"),
    Input(source="app", property="controlState.selectedDataset2Id"),
    Input(source="app", property="controlState.selectedVariable2Name"),
    Input(source="app", property="controlState.selectedTimeLabel"),
    Input(source="app", property="controlState.selectedPlaceGeometry"),
    Output("button", "disabled"),
)
def update_button(
    ctx: Context,
    dataset_id_1: str | None = None,
    variable_name_1: str | None = None,
    dataset_id_2: str | None = None,
    variable_name_2: str | None = None,
    time_label: float | None = None,
    place_geometry: str | None = None,
) -> altair.Chart | None:
    enabled = all(
        (
            dataset_id_1,
            dataset_id_2,
            variable_name_1,
            variable_name_2,
            time_label,
            place_geometry,
        )
    )
    return not enabled
