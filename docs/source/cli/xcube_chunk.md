# `xcube chunk`

(Re-)chunk dataset.

```bash
$ xcube chunk --help
```
    
    Usage: xcube chunk [OPTIONS] <input> <output>
    
      (Re-)chunk dataset. Changes the external chunking of all variables of
      <input> according to <chunks> and writes the result to <output>.
    
    Options:
      -f, --format <format>  Format of the output. If not given, guessed from
                             <output>.
      -p, --params <params>  Parameters specific for the output format. Comma-
                             separated list of <key>=<value> pairs.
      -c, --chunks <chunks>  Chunk sizes for each dimension. Comma-separated list
                             of <dim>=<size> pairs, e.g. "time=1,lat=270,lon=270"
      --help                 Show this message and exit.


Example:

    $ xcube chunk input_not_chunked.zarr output_rechunked.zarr --chunks "time=1,lat=270,lon=270"
