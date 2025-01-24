## Server-side UI Contributions

Starting with xcube Server 1.8 and xcube Viewer 1.4 it is possible to enhance 
the viewer UI by _server-side contributions_ programmed in Python.
For this to work, service providers can now configure xcube Server to load 
one or more Python modules that provide UI-contributions of type 
`xcube.webapi.viewer.contrib.Panel`.
Users can create `Panel` objects and use the two decorators 
`layout()` and `callback()` to implement the UI and the interaction 
behaviour, respectively. The new functionality is provided by the
[Chartlets](https://bcdev.github.io/chartlets/) Python library.

A working example can be found in the 
[xcube repository](https://github.com/xcube-dev/xcube/tree/5ebf4c76fdccebdd3b65f4e04218e112410f561b/examples/serve/panels-demo).

## Contributions

The following contributions are in use by this instance of xcube Viewer:

${extensions}

## Available State Properties

In the following, xcube Viewer's state properties are listed.
These properties can be accessed in the input and state channels you pass
to the `layout()` and `callback()` decorators. To trigger a callback 
call when a state property changes use the input syntax
`Input("@app", <property>)`. To just read a property from the state use 
`State("@app", <property>)`.

${derivedState}
