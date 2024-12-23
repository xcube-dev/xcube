import xarray as xr
from xcube.core.gridmapping import GridMapping
import numpy as np
import pyproj


def create_s2plus_dataset():
    x = xr.DataArray(
        [310005.0, 310015.0, 310025.0, 310035.0, 310045.0],
        dims=["x"],
        attrs=dict(units="m", standard_name="projection_x_coordinate"),
    )
    y = xr.DataArray(
        [5689995.0, 5689985.0, 5689975.0, 5689965.0, 5689955.0],
        dims=["y"],
        attrs=dict(units="m", standard_name="projection_y_coordinate"),
    )
    lon = xr.DataArray(
        [
            [0.272763, 0.272906, 0.273050, 0.273193, 0.273336],
            [0.272768, 0.272911, 0.273055, 0.273198, 0.273342],
            [0.272773, 0.272917, 0.273060, 0.273204, 0.273347],
            [0.272779, 0.272922, 0.273066, 0.273209, 0.273352],
            [0.272784, 0.272927, 0.273071, 0.273214, 0.273358],
        ],
        dims=["y", "x"],
        attrs=dict(units="degrees_east", standard_name="longitude"),
    )
    lat = xr.DataArray(
        [
            [51.329464, 51.329464, 51.329468, 51.32947, 51.329475],
            [51.329372, 51.329376, 51.32938, 51.329384, 51.329388],
            [51.329285, 51.329285, 51.32929, 51.329292, 51.329296],
            [51.329193, 51.329197, 51.32920, 51.329205, 51.329205],
            [51.329100, 51.329105, 51.32911, 51.329113, 51.329117],
        ],
        dims=["y", "x"],
        attrs=dict(units="degrees_north", standard_name="latitude"),
    )
    rrs_443 = xr.DataArray(
        [
            [0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
            [0.014000, 0.014000, 0.016998, 0.016998, 0.016998],
            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
            [0.019001, 0.019001, 0.016998, 0.016998, 0.016998],
        ],
        dims=["y", "x"],
        attrs=dict(units="sr-1", grid_mapping="transverse_mercator"),
    )
    rrs_665 = xr.DataArray(
        [
            [0.025002, 0.019001, 0.008999, 0.012001, 0.022999],
            [0.028000, 0.021000, 0.009998, 0.008999, 0.022999],
            [0.036999, 0.022999, 0.007999, 0.008999, 0.023998],
            [0.041000, 0.022999, 0.007000, 0.009998, 0.021000],
            [0.033001, 0.018002, 0.007999, 0.008999, 0.021000],
        ],
        dims=["y", "x"],
        attrs=dict(units="sr-1", grid_mapping="transverse_mercator"),
    )
    transverse_mercator = xr.DataArray(
        np.array([0xFFFFFFFF], dtype=np.uint32),
        attrs=dict(
            grid_mapping_name="transverse_mercator",
            scale_factor_at_central_meridian=0.9996,
            longitude_of_central_meridian=3.0,
            latitude_of_projection_origin=0.0,
            false_easting=500000.0,
            false_northing=0.0,
            semi_major_axis=6378137.0,
            inverse_flattening=298.257223563,
        ),
    )
    return xr.Dataset(
        dict(rrs_443=rrs_443, rrs_665=rrs_665, transverse_mercator=transverse_mercator),
        coords=dict(x=x, y=y, lon=lon, lat=lat),
        attrs={
            "title": "T31UCS_20180802T105621",
            "conventions": "CF-1.6",
            "institution": "VITO",
            "product_type": "DCS4COP Sentinel2 Product",
            "origin": "Copernicus Sentinel Data",
            "project": "DCS4COP",
            "time_coverage_start": "2018-08-02T10:59:38.888000Z",
            "time_coverage_end": "2018-08-02T10:59:38.888000Z",
        },
    )


GEO_CRS = pyproj.crs.CRS(4326)
dataset = create_s2plus_dataset()

gm = GridMapping.from_dataset(dataset, prefer_is_regular=False, tolerance=1e-6)
# Should pick the geographic one which is irregular
assert "Geographic" in gm.crs.type_name
assert not gm.is_regular.compute()

gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS)
# Should pick the geographic one which is irregular
assert "Geographic" in gm.crs.type_name
assert gm.is_regular is False

gm = GridMapping.from_dataset(dataset, prefer_crs=GEO_CRS, prefer_is_regular=True)
# Should pick the geographic one which is irregular
assert "Geographic" in gm.crs.type_name
assert gm.is_regular is False
