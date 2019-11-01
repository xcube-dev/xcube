### package renaming on master

**problem**: package renaming on master complicates merging of 0.2.x changes

**solution**: lets release a 0.2, then a 0.3 soon. 

### xcube.api and xcube.util

**problem**: importing xcube.api imports *all* xcube 
dependencies and all xcube API code - this is slow and destroys our 
original idea of modularisation!
Often a component from xcube.api is required in xcube.util module, but
for xcube.util we force to only import what is needed. So we moved already 
a lot of xcube.api functions into xcube.util but that is actually 
the wrong place. Plugins may need to import lightweight parts of the 
xcube API before they import heavy packages. 
What is API or not should be defined by the docs, 
not by a package.  

**solution**: restructure xcube packages once more:
  * xcube.core - rename from xcube.api
  * xcube.cli - no changes
  * xcube.webapi - no changes
  * xcube.plugins - no changes
  * xcube.util - move modules mainly dealing with xarray datasets to xcube.core 
    
**example**: 

Consider a plugin's `plugin` module which should execute in milliseconds:

This is bad, `xcube.api` loads all dependencies:

    import xcube.api
    
    def init_plugin(ext_registry: xcube.api.ExtensionRegistry):
        ...

This is good, `xcube.core` doesn't do anything, but contains sub-modules that are API:

    import xcube.core.ext
    
    def init_plugin(ext_registry: xcube.core.ext.ExtensionRegistry):
        ...
        
        
### edit_metadata

**problem**: 
- API function does not conform to guidelines.
- CLI `--metadata` option description not clear 
**solution**:
- Rename `edit_metadata` to `update_cube_metadata`
- Move to `xcube.core.update` module
- Update help for CLI `--metadata` option
- Rename CLI command to `update` (?)