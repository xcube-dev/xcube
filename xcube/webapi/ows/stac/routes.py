# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

from xcube.server.api import ApiHandler, ApiError

from .api import api
from .context import StacContext
from .controllers import (
    get_collection,
    get_collection_queryables,
    get_single_collection_items,
    get_collection_schema,
)
from .controllers import get_collection_item
from .controllers import get_datasets_collection_items
from .controllers import get_collections
from .controllers import get_conformance
from .controllers import get_root
from .controllers import search
from .config import (
    PATH_PREFIX,
    DEFAULT_COLLECTION_ID,
)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX, slash=True)
class CatalogRootHandler(ApiHandler[StacContext]):
    @api.operation(
        operation_id="getCatalogRoot", summary="Get the STAC catalog's root."
    )
    async def get(self):
        result = get_root(self.ctx.datasets_ctx, self.request.reverse_base_url)
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/conformance", slash=True)
class CatalogConformanceHandler(ApiHandler[StacContext]):
    @api.operation(
        operation_id="getCatalogConformance",
        summary="Get the STAC catalog's conformance.",
    )
    async def get(self):
        result = get_conformance()
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections", slash=True)
class CatalogCollectionsHandler(ApiHandler[StacContext]):
    @api.operation(
        operation_id="getCatalogCollections",
        summary="Get the STAC catalog's collections.",
    )
    async def get(self):
        result = get_collections(self.ctx, self.request.reverse_base_url)
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections/{collectionId}", slash=True)
class CatalogCollectionHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="getCatalogCollection", summary="Get a STAC catalog collection."
    )
    async def get(self, collectionId: str):
        result = get_collection(self.ctx, self.request.reverse_base_url, collectionId)
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections/{collectionId}/items", slash=True)
class CatalogCollectionItemsHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="getCatalogCollectionItems",
        summary="Get the items of a STAC catalog collection.",
    )
    async def get(self, collectionId: str):
        get_items_args = dict(
            ctx=self.ctx.datasets_ctx,
            base_url=self.request.reverse_base_url,
            collection_id=collectionId,
        )
        if collectionId == DEFAULT_COLLECTION_ID:
            if "limit" in self.request.query:
                get_items_args["limit"] = self.request.get_query_arg("limit", type=int)
            if "cursor" in self.request.query:
                get_items_args["cursor"] = self.request.get_query_arg(
                    "cursor", type=int
                )
            result = get_datasets_collection_items(**get_items_args)
        else:
            result = get_single_collection_items(**get_items_args)

        return await self.response.finish(result, content_type="application/geo+json")


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections/{collectionId}/items/{featureId}", slash=True)
class CatalogCollectionItemHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="getCatalogCollectionItem",
        summary="Get an item of a STAC catalog collection.",
    )
    async def get(self, collectionId: str, featureId: str):
        result = get_collection_item(
            self.ctx.datasets_ctx,
            self.request.reverse_base_url,
            collectionId,
            featureId,
        )
        return await self.response.finish(result, content_type="application/geo+json")


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/search", slash=True)
class CatalogSearchHandler(ApiHandler[StacContext]):
    @api.operation(
        operation_id="searchCatalogByKeywords",
        summary="Search the STAC catalog by keywords.",
    )
    async def get(self):
        # TODO (forman): get search params from query
        result = search(self.ctx.datasets_ctx, self.request.reverse_base_url)
        return await self.response.finish(result)

    @api.operation(
        operation_id="searchCatalogByJSON",
        summary="Search the STAC catalog by JSON request body.",
    )
    async def post(self):
        # TODO (forman): get search params from request body
        result = search(self.ctx.datasets_ctx, self.request.reverse_base_url)
        return await self.response.finish(result)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections/{collectionId}/queryables", slash=True)
class QueryablesHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="queryables",
        summary="Return a JSON Schema defining the supported "
        'metadata filters (also called "queryables") '
        "for a specific collection.",
    )
    async def get(self, collectionId):
        schema = get_collection_queryables(
            ctx=self.ctx.datasets_ctx, collection_id=collectionId
        )
        return await self.response.finish(schema)


# noinspection PyAbstractClass,PyMethodMayBeStatic
@api.route(PATH_PREFIX + "/collections/{collectionId}/schema", slash=True)
class SchemaHandler(ApiHandler[StacContext]):
    # noinspection PyPep8Naming
    @api.operation(
        operation_id="schema",
        summary="Return a JSON Schema defining the range "
        "(i.e. data variables) of a specific collection.",
    )
    async def get(self, collectionId):
        schema = get_collection_schema(
            ctx=self.ctx.datasets_ctx,
            base_url=self.request.reverse_base_url,
            collection_id=collectionId,
        )
        return await self.response.finish(schema)
