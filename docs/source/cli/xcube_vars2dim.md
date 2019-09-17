## `xcube vars2dim`

Convert cube variables into new dimension.

```bash
$ xcube vars2dim --help
```
    
    Usage: xcube vars2dim [OPTIONS] CUBE
    
      Convert cube variables into new dimension. Moves all variables of CUBE
      into into a single new variable <var-name> with a new dimension <DIM-NAME>
      and writes the results to <OUTPUT>.
    
    Options:
      -v, --variable, --var <VARIABLE>
                                      Name of the new variable that includes all
                                      variables. Defaults to "data".
      -d, --dim_name <DIM-NAME>       Name of the new dimension into variables.
                                      Defaults to "var".
      -o, --output <OUTPUT>           Output path. If omitted,
                                      '<INPUT>-vars2dim.<INPUT-FORMAT>' will be
                                      used.
      -f, --format <FORMAT>           Format of the output. If not given, guessed
                                      from <OUTPUT>.
      --help                          Show this message and exit.


