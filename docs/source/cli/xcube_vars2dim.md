## `xcube vars2dim`

Convert cube variables into new dimension.

```bash
    $ xcube vars2dim --help
```
    
    Usage: xcube vars2dim [OPTIONS] <cube>
    
      Convert cube variables into new dimension. Moves all variables of <cube>
      into into a single new variable <var-name> with a new dimension <dim-name>
      and writes the results to <output>.
    
    Options:
      -d, --dim_name <dim-name>  Name of the new dimension into variables.
                                 Defaults to "var".
      -v, --var_name <var-name>  Name of the new variable that includes all
                                 variables. Defaults to "data".
      -o, --output <output>      Output file.
      -f, --format <format>      Format of the output. If not given, guessed from
                                 <output>.
      --help                     Show this message and exit.


