### API "edit_metadata", CLI "xcube edit"

- API function `xcube.core.edit.edit_metadata` 
    * name is not consistent with other names
    * signature not consistent, works on local file paths 
      rather than `xr.Dataset` objects
- Make consistent with similar existing update functions
    * Rename `edit_metadata` to `update_cube_metadata` (?)
    * Move to `xcube.core.update` module
    * Rename CLI command to `xcube update` (?)
- API function `metadata_path` arg and CLI `--metadata` option description 
  not clear, explain expected format, enhance doc/help string 

### Restructuring

* fix time constants in `xcube.core.timeccord` and `xcube.core.new`, move to `xcube.constants`
* move resampling defaults from `xcube.core.reproject` to `xcube.defaults`
* move defaults from `xcube.core.extract` to `xcube.defaults`