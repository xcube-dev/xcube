import altair
from chartlets import Component, Input, State, Output
from chartlets.components import Box, Button, Plot, Select

from xcube.webapi.viewer.contrib import Panel
from xcube.webapi.viewer.contrib import get_dataset
from xcube.server.api import Context


panel = Panel(__name__, title="2D Histogram")


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
            "gap": "6px",
        },
    )

    var_names, var_name_1, var_name_2 = get_var_select_options(dataset)

    select_var_1 = Select(
        id="select_var_1",
        label="Variable 1",
        value=var_name_1,
        options=var_names
    )
    select_var_2 = Select(
        id="select_var_2",
        label="Variable 2",
        value=var_name_2,
        options=var_names
    )

    select_box = Box(
        children=[select_var_1, select_var_2],
        style={
            "display": "flex",
            "flexDirection": "row",
            "gap": 6
        }
    )

    button = Button(
        id="button",
        text="Update",
        disabled=True,
        style={"maxWidth": 100}
    )

    return Box(
        children=[plot, select_box, button],
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
) -> altair.Chart | None:
    dataset = get_dataset(ctx, dataset_id)
    if dataset is None or place_geometry is None:
        print("panel disabled")
        return None
    variable_1 = dataset[var_name_1]
    variable_2 = dataset[var_name_2]
    print("variable_1", variable_1)
    print("variable_2", variable_2)
    print("time_label", time_label)
    print("place_geometry", place_geometry)
    return None


@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    Input(source="app", property="controlState.selectedPlaceGeometry"),
    Output("button", "disabled"),
)
def enable_button(
    ctx: Context,
    dataset_id: str | None = None,
    place_geometry: str | None = None,
) -> bool:
    dataset = get_dataset(ctx, dataset_id)
    return dataset is None or place_geometry is None


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
