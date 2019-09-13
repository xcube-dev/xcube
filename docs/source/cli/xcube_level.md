# `xcube level`

Generate multi-resolution levels.

```bash
$ xcube level --help
```
    
    Usage: xcube level [OPTIONS] <input>
    
      Generate multi-resolution levels. Transform the given dataset by <input>
      into the levels of a multi-level pyramid with spatial resolution
      decreasing by a factor of two in both spatial dimensions and write the
      result to directory <output>.
    
    Options:
      -o, --output <output>           Output directory. If omitted,
                                      "<input>.levels" will be used.
      -l, --link                      Link the <input> instead of converting it to
                                      a level zero dataset. Use with care, as the
                                      <input>'s internal spatial chunk sizes may
                                      be inappropriate for imaging purposes.
      -t, --tile-size <tile-size>     Tile size, given as single integer number or
                                      as <tile-width>,<tile-height>. If omitted,
                                      the tile size will be derived from the
                                      <input>'s internal spatial chunk sizes. If
                                      the <input> is not chunked, tile size will
                                      be 512.
      -n, --num-levels-max <num-levels-max>
                                      Maximum number of levels to generate. If not
                                      given, the number of levels will be derived
                                      from spatial dimension and tile sizes.
      --help                          Show this message and exit.

    
Example:

    $ xcube level -l -t 720 data/cubes/test-cube.zarr

