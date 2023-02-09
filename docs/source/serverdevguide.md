
## xcube Server Framework

The _xcube server framework_ has been introduced in xcube 0.13.
It is implemented in the `xcube.server` module.

The main features of the framework are its modularity and extendability.
A server comprises one or more _server APIs_, and each API contributes 
dedicated _API routes_. An API route defines server's endpoint 
and implements one or more of the HTTP request's `GET`, `PUT`, `DELETE`, 
etc., methods.

The xcube server framework is developed with independence 
of a concrete web server in mind. However, the xcube server default web server 
is [Tornado](https://www.tornadoweb.org/), but stubs are already provided 
for [Flask](https://flask.palletsprojects.com/).

### Server API

A _server API_ is a pluggable server extension that can be provided
by a xcube plugin. Also, the xcube server's standard APIs, such as 
`datasets`, `places`, `tiles`, `ows.wmts` located in the 
`xcube.webapi` module are server API extensions that are registered 
in the `xcube.plugin` module. 

A server API can have its own server configuration. 
It describes its configuration by a [JSON Schema](https://json-schema.org/),
so it can be validated by the server.


### Server API extension module

A server API extension module is any Python module that exports a variable,
typically named `api`, whose value is an instance of the 
class `xcube.server.Api`.

We recommend laying out the API module as a sub-package with the following 
structure: 

```text
<api-module>/
  |- __index__.py     # Export the API object: from .api import api 
  |- api.py           # Define the API object: api = xcube.server.Api(...)
  |- routes.py        # Optional: Add API's routes and implements handlers 
  |- config.py        # Optional: Define the configuration's JSON schema 
  |- context.py       # Optional: Implement access to API resources
  |- controllers.py   # Optional: Implement the logic of the route handlers
```

The name of the API extension module must be registered in xcube.
API extensions are registered using the xcube extension point named
`"xcube.server.api"`, also defined by
`xcube.constants.EXTENSION_POINT_SERVER_APIS`.

By xcube convention, the name of a xcube plugin package starts with `xcube_`.
And also by xcube convention, the registration of xcube extensions is done 
by a function called `init_plugin` that is defined in top-level module 
named `plugin`.

Here is an example of a xcube plugin package `xcube_ew4dv` that implements
its server API module in `xcube_ew4dv.webapi`. Here is the contents of 
the `xcube_ew4dv.plugin` module (file `xcube_ew4dv/plugin.py`):

```python
from xcube.constants import EXTENSION_POINT_SERVER_APIS
from xcube.util.extension import ExtensionRegistry
from xcube.util.extension import import_component

def init_plugin(ext_registry: ExtensionRegistry):
    ext_registry.add_extension(
        loader=import_component("xcube_ew4dv.webapi:api"),
        point=EXTENSION_POINT_SERVER_APIS,
        name="Earthwave 4D Viewer API"
    )    
```

---
You may have noticed that there is no need to import any component
directly from `xcube_ew4dv.webapi`. This is done by design - we defer
importing of extension code until it is really needed, e.g., when the
xcube server is started using CLI tool `xcube serve`. Importing modules
eagerly may take up to a few seconds, which may be already too much if 
you just want to call `xcube --help` or `xcube serve --help`.
---

Since in the above example the package is called `xcube_ew4dv`, xcube will
automatically recognise it as a potential xcube plugin package because of 
the name prefix `xcube_`. If any of the following conditions 

- registration function is called `init_plugin` and
- registration function is in top-level module `plugin` and
- package name starts with `xcube_`

cannot be satisfied, then it must be registered in the package's entry points.

If you use `setup.py` and `setuptools` in your project folder:

```python
from setuptools import setup

setup(
    # ...
    entry_points={
        'xcube_plugins': [
            # This is xcube convention (no need to specify it here at all): 
            # 'xcube_ew4dv = xcube_ew4dv.plugin:init_plugin',
            # If you differ, specify your init_plugin() here:
            'xcube_ew4dv = ew4dv.xcube:register_api',
        ],
    }
)
```

If you use `setup.cfg` in your project folder:

TODO - add the above for `setup.cfg`

### API definition (`api.py`) 

In its simplest form an API definition just requires a unique API identifier: 

```python
from xcube.server.api import Api

api = Api("4d-viewer")
```

Many APIs may be configurable through the xcube server configuration.
Then we can pass a JSON schema for the specific API configuration,
which is defined our `config.py`:

```python
from xcube.server.api import Api
from .config import CONFIG_SCHEMA

api = Api("4d-viewer",
          config_schema=CONFIG_SCHEMA)
```

If your API manages state, for example it could maintain caches for frequently
requested resources, this state can be kept in a dedicated API context
object. Then we can pass a factory for a specific API context object
which is defined our `context.py`. In our case it is a class `FourDContext`:

```python
from xcube.server.api import Api
from .config import CONFIG_SCHEMA
from .context import FourDContext

api = Api("4d-viewer",
          config_schema=CONFIG_SCHEMA,
          create_ctx=FourDContext)
```

Both the configuration and the context object are accessible
from your API's route handlers. We'll see how that works in a moment.

Your API extension may be build on top of other APIs and may want to share
their configuration and context information. We can tell our API to
require other APIs. In the following case, the API depends on 
the "datasets" and "tiles" APIs. That means, the `FourDContext` object
will have access to the context objects of the "datasets" and "tiles" APIs:

```python
api = Api("4d-viewer",
          config_schema=CONFIG_SCHEMA,
          create_ctx=FourDContext,
          required_apis=["datasets", "tiles"])
```

The next section describes how to add routes to the API.
But note that it is not required for an API to have any routes.
An API might go without any routes and just provide a server middleware 
(e.g. do something on any request) or serve as a base API for 
other dependent APIs.

### API Routes (`routes.py`)

An API route defines server's endpoint and implements one or more of the 
HTTP request's `GET`, `PUT`, `DELETE`, etc., methods.

The following route implements an asynchronous `GET` handler for the 
endpoint `/tiles4d/{datasetId}/{varName}/{z}/{y}/{x}`. We can also implement
handlers for other HTTP methods in the same class. 

```python
from xcube.server.api import ApiHandler
from .api import api
from .context import FourDContext

@api.route("/tiles4d/{datasetId}/{varName}/{z}/{y}/{x}")
class FourDTileHandler(ApiHandler[FourDContext]):
  
    @api.operation(operation_id="getFourDTile",
                   summary="Get a tile for the 4D Viewer.")
    async def get(self, 
                  datasetId: str, 
                  varName: str, 
                  z: str, 
                  y: str, 
                  x: str):
       ctx = self.ctx        # State/resource access of type FourDContext.
       config = ctx.config   # Server configuration as dict.
       request = self.request    # The request
       response = self.response  # The response
       # Implementation ...
       await response.finish()
```

We can now run `xcube serve` and open <http://localhost:8080/openapi.html>
in a browser. We get:

TODO: add screenshot

### API Configuration (`config.py`)

### API Context (`context.py`)

### API Controllers (`controllers.py`)


### TODO - describe more API details

- Can provide zero, one, or more API handlers
- Must describe its API via OpenAPI declarations, if any
- Can act as server middleware (e.g. do something on any request)
- Can have its own configuration 
- Must describe its configuration, if any, by JSON schema
- Can create its own context object 
- Has a unique name
- Can depend on other extensions, by name
- Can access configurations of other extensions, by name
- Can access context object of other extensions, by name

### TODO - describe server + API config

- Server configuration
  * = server base configuration + server extension configurations.
  * It is a JSON object.
- Server base configuration
  * Configuration items are properties of the server JSON object.
  * Properties: version number, address and port, authentication, 
    authorisation, server metadata…
- Server extension configuration
  * Is a property of the server JSON object (using the extension’s name).
  * Configuration may be of any JSON type as required by the server extension.
- Server configuration JSON schema 
  * = server base configuration JSON schema + server extension 
  configuration JSON schemas.

### TODO - describe API context object

- Provide the runtime state of a server extension.
- Are created and updated from a server configuration changes.
- Are optional. Some server extensions don’t require a context object.
  By default, the context object is null. 
- Typically, provide access (= Python API) to the resources served by the 
  handlers of the extension (= REST API).
- The context’s Python API is used by the extension itself but can also be 
  used by dependent extensions.
