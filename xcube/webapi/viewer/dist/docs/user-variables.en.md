A _user variable_ is a variable that is defined by a _name_, _title_, _units_, 
and by an algebraic _expression_ that is used to compute the variable's array 
data. User variables are added to the currently selected dataset and their
expressions are evaluated in the context of the selected dataset.

**Name**: A name that is unique within the selected dataset's variables.
The name must start with a letter optionally followed by letters or digits. 

**Title**: Optional display name of the variable in the user interface.

**Units**: Optional physical units of the computed data values.
For example, units are used to group time-series.

**Expression**: An algebraic expression used to compute the variable's data
values. The syntax is that of [Python expressions](https://docs.python.org/3/reference/expressions.html). 
The expression may reference the following names:
- the current dataset's data variables;
- the numpy constants `e`, `pi`, `nan`, `inf`;
- all [numpy ufunc](https://numpy.org/doc/stable/reference/ufuncs.html) 
  functions;
- the [`where`](https://docs.xarray.dev/en/stable/generated/xarray.where.html) function.

The majority of Python numerical and logical operators are supported,
however, the logical operators `and`, `or`, and `not` cannot be used with 
array variables as they require boolean values as operands. Use the bitwise
operators `&`, `|`, `~` instead or use the
corresponding functions `logical_and()`, `logical_or()`, and `logical_not()`.
Python built-in functions such as `min()` and `max()` are not supported,
use `fmin()` and `fmax()` instead.

Expression examples:

- Mask out where a variable `chl` is lower than zero: `where(chl >= 0, chl, nan)`
- Sentinel-2 vegetation index or NDVI: `(B08 - B04) / (B08 + B04)`
- Sentinel-2 moisture index: `(B8A - B11) / (B8A + B11)`

Invalid expressions return an error message.

CTRL+SPACE: activates the autocomplete feature, which lists 
available Python functions and constants