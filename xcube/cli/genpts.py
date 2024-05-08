# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import click

DEFAULT_OUTPUT_PATH = "out.csv"
DEFAULT_NUM_POINTS_MAX = 100


@click.command(name="genpts", hidden=True)
@click.argument("cube")
@click.option(
    "--output",
    "-o",
    "output_path",
    help=f'Output path. Defaults to "{DEFAULT_OUTPUT_PATH}".',
    default=DEFAULT_OUTPUT_PATH,
)
@click.option(
    "--number",
    "-n",
    "num_points_max",
    help=f'Number of point_data. Defaults to "{DEFAULT_NUM_POINTS_MAX}".',
    type=int,
    default=DEFAULT_NUM_POINTS_MAX,
)
def genpts(cube: str, output_path: str, num_points_max: int):
    """Generate synthetic data points from CUBE.

    Generates synthetic data points for a given data CUBE and write points to a CSV and GeoJSON file.
    The primary use of the tool is to point datasets for machine learning tasks
    and to create test points for the "xcube extract" command.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    def _all_data_vars_valid(dataset: xr.DataArray):
        for _, var in dataset.data_vars.items():
            if not np.isfinite(var):
                return False
        return True

    cube = xr.open_zarr(cube)

    num_points = 0
    point_data = dict(time=[], lat=[], lon=[])
    for var_name in cube.data_vars:
        point_data[var_name] = []

    spatial_res = ((cube.lon[-1] - cube.lon[0]) / (cube.lon.size - 1)).values
    temporal_res = ((cube.time[-1] - cube.time[0]) // (cube.time.size - 1)).values

    while num_points < num_points_max:
        it = np.random.randint(0, cube.time.size)
        iy = np.random.randint(0, cube.lat.size)
        ix = np.random.randint(0, cube.lon.size)

        point = cube.isel(time=it, lat=iy, lon=ix)

        if _all_data_vars_valid(point):
            point_data["time"].append(
                point.time.values + 0.5 * temporal_res * np.random.logistic(scale=0.2)
            )
            point_data["lat"].append(
                point.lat.values + 0.5 * spatial_res * np.random.logistic(scale=0.2)
            )
            point_data["lon"].append(
                point.lon.values + 0.5 * spatial_res * np.random.logistic(scale=0.2)
            )
            for var_name, var in point.data_vars.items():
                value = var.values
                if np.issubdtype(value.dtype, np.floating):
                    value += np.random.logistic(scale=0.05)
                point_data[var_name].append(value)
            num_points += 1

    if output_path.endswith(".csv"):
        pd.DataFrame(point_data).to_csv(output_path)
    elif output_path.endswith(".geojson"):
        import json

        with open(output_path, "w") as fp:
            json.dump(_to_geojson_dict(point_data, num_points), fp, indent=2)


def _to_geojson_dict(point_data, num_points):
    import numpy as np

    features = []
    for i in range(num_points):
        x = float(point_data["lon"][i])
        y = float(point_data["lat"][i])
        properties = {}
        for k, v in point_data.items():
            if k in {"lon", "lat"}:
                continue
            value = v[i]
            if np.issubdtype(value.dtype, np.floating):
                value = float(value)
            elif np.issubdtype(value.dtype, np.datetime64):
                value = str(value)
            else:
                value = int(value)
            properties[k] = value
        features.append(
            {
                "type": "Feature",
                "id": i,
                "geometry": {
                    "type": "Point",
                    "coordinates": (x, y),
                },
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    genpts.main()
