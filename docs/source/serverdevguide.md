
## xcube Server Framework

The _xcube server framework_ has been introduced in xcube 0.13.
It is implemented in the `xcube.server` module.

The main feature of the framework is its modularity and extendability.
A server comprises one or more _server APIs_, and each API contributes 
dedicated _API routes_. An API route defines server's endpoint 
and implements one or more of the HTTP request's `GET`, `PUT`, `DELETE`, 
etc., methods.

The xcube server framework is developed with independence 
of a concrete web server in mind. However, the xcube server default web server 
is [Tornado](https://www.tornadoweb.org/), but stubs are already provided 
for [Flask](https://flask.palletsprojects.com/).

### The Server API Extension

A _server API_ is a pluggable server extension that can be provided
by a xcube plugin. Also, the xcube server's standard APIs, such as 
`datasets`, `places`, `tiles`, `ows.wmts` located in the 
`xcube.webapi` module are server API extensions that are registered 
in the `xcube.plugin` module. 

A server API can have its own server configuration. 
It describes its configuration by a [JSON Schema](https://json-schema.org/),
so it can be validated by the server.



### Anatomy of an API extension module

An API extension module is any Python module that exports a
variable named `api` whose value is an instance of the 
class `xcube.server.Api`. 

We recommend laying out the API module as a sub-package with the following 
structure: 

```text
<api-module>/
  |- __index__.py     # Export the API object: from .api import api 
  |- api.py           # Define the API object: api = xcube.server.Api(...)
  |- routes.py        # Optional: Add API's routes and implements handlers 
  |- context.py       # Optional: Implement access to API resources
  |- controllers.py   # Optional: Implement the logic of the routes
  |- config.py        # Optional: Define the configuration's JSON schema 
```

The name of the API extension module must be registered in the `plugin`
module of your main package. The registration is done in a function 
that must be called `init_plugin`. API extensions are registered
using the extension point named `"xcube.server.api"`, 
also defined by `xcube.constants.EXTENSION_POINT_SERVER_APIS`.

Here is an example of a xcube plugin `xcube_ew4dv` that implements
its server API module in `xcube_ew4dv.webapi`:

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

You may have noticed that there is no need to import any component
directly from `xcube_ew4dv.webapi`. This is done by design - we defer
importing of extension code until it is really needed, e.g., when the
xcube server is started using CLI tool `xcube serve`. Importing modules
eagerly may take up to a few seconds, which may be already too much if 
you just want to call `xcube --help` or `xcube serve --help`.

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
- Are created and updated from a server extension’s configuration.
- Are optional. Some server extensions don’t require a context object.
  By default, the context object is null. 
- Typically, provide access (= Python API) to the resources served by the 
  handlers of the extension (= REST API).
- The context’s Python API is used by the extension itself but can also be 
  used by dependent extensions.
