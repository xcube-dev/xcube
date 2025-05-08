from xcube.core.store import new_data_store
from xcube.core.resampling.reproject import reproject_dataset
from xcube.core.resampling import resample_in_space
from xcube.core.gridmapping import GridMapping
from xcube.core.chunk import chunk_dataset
from datetime import datetime
import matplotlib.pyplot as plt
import pyproj
import numpy as np


# store = new_data_store("s3", root="deep-esdl-public")
# mlds_lc = store.open_data("LC-1x2025x2025-2.0.0.levels")
# ds = mlds_lc.base_dataset
#
# ds = ds.sel(
#     time=slice(datetime(2020, 1, 1), datetime(2022, 1, 1)),
#     lat=slice(60, 40),
#     lon=slice(0, 20),
# )
# print(ds)
# ds = chunk_dataset(
#     ds,
#     dict(time=1, lat=1000, lon=1000, bounds=2),
#     format_name="zarr",
# )
# print(ds)
#
# store_file = new_data_store("file")
# store_file.write_data(ds, "land_cover.zarr", replace=True)
#
# print("done")

store_file = new_data_store("file")
ds = store_file.open_data("land_cover.zarr")

bbox = [0, 40, 20, 60]
target_crs = "EPSG:3035"
t = pyproj.Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
target_bbox = t.transform_bounds(*bbox)
spatial_res = 300
x_size = int((target_bbox[2] - target_bbox[0]) / spatial_res) + 1
y_size = int(abs(target_bbox[3] - target_bbox[1]) / spatial_res) + 1
target_gm = GridMapping.regular(
    size=(x_size, y_size),
    xy_min=(target_bbox[0] - spatial_res / 2, target_bbox[1] - spatial_res / 2),
    xy_res=spatial_res,
    crs=target_crs,
    tile_size=1000,
)

# new reproject
start = datetime.now()
ds_reproject = reproject_dataset(ds, target_gm=target_gm)
elapsed = (datetime.now() - start).total_seconds()
print(f"Computational time reproject_dataset: {elapsed:.2f}sec")
print(ds_reproject)

# start = datetime.now()
# ds_reproject.lccs_class.isel(time=-1)[::5, ::5].plot()
# elapsed = (datetime.now() - start).total_seconds()
# print(f"Computational time plot: {elapsed:.2f}sec")
# plt.show()
