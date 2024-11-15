from chartlets import Component, Input, State, Output
from chartlets.components import Box, Dropdown, Checkbox, Typography

from xcube.webapi.viewer.contrib import Panel
from xcube.webapi.viewer.contrib import get_datasets_ctx
from xcube.server.api import Context


panel = Panel(__name__, title="Panel B")


COLORS = [("red", 0), ("green", 1), ("blue", 2), ("yellow", 3)]


@panel.layout(
    Input(source="app", property="controlState.selectedDatasetId"),
)
def render_panel(
    ctx: Context,
    dataset_id: str = "",
) -> Component:

    opaque = False
    color = 0

    opaque_checkbox = Checkbox(
        id="opaque",
        value=opaque,
        label="Opaque",
    )

    color_dropdown = Dropdown(
        id="color",
        value=color,
        label="Color",
        options=COLORS,
        style={"flexGrow": 0, "minWidth": 80},
    )

    info_text = Typography(
        id="info_text", text=update_info_text(ctx, dataset_id, opaque, color)
    )

    return Box(
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",
            "gap": "6px",
        },
        children=[opaque_checkbox, color_dropdown, info_text],
    )


# noinspection PyUnusedLocal
@panel.callback(
    Input(source="app", property="controlState.selectedDatasetId"),
    Input("opaque"),
    Input("color"),
    State("info_text", "text"),
    Output("info_text", "text"),
)
def update_info_text(
    ctx: Context,
    dataset_id: str = "",
    opaque: bool = False,
    color: int = 0,
    info_text: str = "",
) -> str:
    ds_ctx = get_datasets_ctx(ctx)
    ds_configs = ds_ctx.get_dataset_configs()

    opaque = opaque or False
    color = color if color is not None else 0
    return (
        f"The dataset is {dataset_id},"
        f" the color is {COLORS[color][0]} and"
        f" it {'is' if opaque else 'is not'} opaque."
        f" The length of the last info text"
        f" was {len(info_text or "")}."
        f" The number of datasets is {len(ds_configs)}."
    )
