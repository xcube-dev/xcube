import logging
from typing import Any, Dict

import shapely.geometry
import shapely.wkt
from shapely.errors import WKTReadingError

from ..context import ServiceContext
from ..errors import ServiceBadRequestError
from ...util.geom import get_dataset_geometry, get_box_split_bounds_geometry
from ...util.perf import measure_time

_LOG = logging.getLogger('xcube')

GeoJsonFeatureCollection = Dict
GeoJsonFeature = Dict


def find_dataset_places(ctx: ServiceContext,
                        place_group_id: str,
                        ds_id: str,
                        query_expr: Any = None,
                        comb_op: str = "and") -> GeoJsonFeatureCollection:
    dataset = ctx.get_dataset(ds_id)
    query_geometry = get_dataset_geometry(dataset)
    return _find_places(ctx,
                        place_group_id,
                        query_geometry=query_geometry,
                        query_expr=query_expr, comb_op=comb_op)


def find_places(ctx: ServiceContext,
                place_group_id: str,
                box_coords: str = None,
                geom_wkt: str = None,
                query_expr: Any = None,
                geojson_obj: Dict = None,
                comb_op: str = "and") -> GeoJsonFeatureCollection:
    query_geometry = None
    if box_coords:
        try:
            query_geometry = get_box_split_bounds_geometry(*[float(s) for s in box_coords.split(",")])
        except (TypeError, ValueError) as e:
            raise ServiceBadRequestError("Received invalid bounding box geometry") from e
    elif geom_wkt:
        try:
            query_geometry = shapely.wkt.loads(geom_wkt)
        except (TypeError, WKTReadingError) as e:
            raise ServiceBadRequestError("Received invalid geometry WKT") from e
    elif geojson_obj:
        try:
            if geojson_obj["type"] == "FeatureCollection":
                query_geometry = shapely.geometry.shape(geojson_obj["places"][0]["geometry"])
            elif geojson_obj["type"] == "Feature":
                query_geometry = shapely.geometry.shape(geojson_obj["geometry"])
            else:
                query_geometry = shapely.geometry.shape(geojson_obj)
        except (IndexError, ValueError, KeyError) as e:
            raise ServiceBadRequestError("Received invalid GeoJSON object") from e
    return _find_places(ctx, place_group_id, query_geometry, query_expr, comb_op)


def _find_places(ctx: ServiceContext,
                 place_group_id: str,
                 query_geometry: shapely.geometry.base.BaseGeometry = None,
                 query_expr: Any = None,
                 comb_op: str = "and") -> GeoJsonFeatureCollection:
    with measure_time() as cm:
        features = __find_places(ctx, place_group_id, query_geometry, query_expr, comb_op)
    _LOG.info(f"{len(features)} places found within {cm.duration} seconds")
    return features


def __find_places(ctx: ServiceContext,
                  place_group_id: str,
                  query_geometry: shapely.geometry.base.BaseGeometry = None,
                  query_expr: Any = None,
                  comb_op: str = "and") -> GeoJsonFeatureCollection:
    if comb_op is not None and comb_op != "and":
        raise NotImplementedError("comb_op not yet supported")
    place_group = ctx.get_place_group(place_group_id)
    if query_geometry is None:
        if query_expr is None:
            return place_group
        else:
            raise NotImplementedError()
    else:
        matching_places = []
        if query_expr is None:
            for place in place_group["features"]:
                geometry = shapely.geometry.shape(place["geometry"])
                if geometry.intersects(query_geometry):
                    matching_places.append(place)
        else:
            raise NotImplementedError()
        return dict(type="FeatureCollection", features=matching_places)
