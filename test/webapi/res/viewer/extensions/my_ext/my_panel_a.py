# Copyright (c) 2018-2025 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from chartlets import Component, Input, Output, State
from chartlets.components import Box, Checkbox, Select, Typography

from xcube.server.api import Context
from xcube.webapi.viewer.contrib import Panel, get_datasets_ctx

panel = Panel(__name__, title="Panel A", position=2, icon="equalizer")


COLORS = [(0, "red"), (1, "green"), (2, "blue"), (3, "yellow")]


@panel.layout(
    State("@app", "selectedDatasetId"),
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

    color_Select = Select(
        id="color",
        value=color,
        label="Color",
        options=COLORS,
        style={"flexGrow": 0, "minWidth": 80},
    )

    info_text = Typography(
        id="info_text", children=[update_info_text(ctx, dataset_id, opaque, color)]
    )

    return Box(
        style={
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",
            "gap": "6px",
        },
        children=[opaque_checkbox, color_Select, info_text],
    )


# noinspection PyUnusedLocal
@panel.callback(
    Input("@app", "selectedDatasetId"),
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
        f" the color is {COLORS[color][1]} and"
        f" it {'is' if opaque else 'is not'} opaque."
        f" The length of the last info text"
        f" was {len(info_text or '')}."
        f" The number of datasets is {len(ds_configs)}."
    )
