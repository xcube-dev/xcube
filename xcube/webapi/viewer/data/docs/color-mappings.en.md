A user-defined color mapping associates data values or ranges of data values 
with color values. The lines in the text box have the general syntax 
`<value>: <color>`, where `<color>` can be

* a list of RGB values, with values in the range 0 to 255, for example,
  `255,165,0` for the color Orange;
* a hexadecimal RGB value, e.g., `#FFA500`;
* or a valid [HTML color name](https://www.w3schools.com/colors/colors_names.asp)
  such as `Orange`, `BlanchedAlmond` or `MediumSeaGreen`.

The color value may be suffixed by a opaqueness (alpha) value in the range
0 to 1, for example `110,220,230,0.5` or `#FFA500,0.8` or `Blue,0`.
Hexadecimal values can also be written including an alpha value,
such as `#FFA500CD`.

The interpretation of the `<value>` depends on the selected color mapping 
type:

* Type **Node**: The color mapping is a linear gradient between the colors 
  with relative distances given by `<value>`, which provide the control 
  values or _nodes_.
* Type **Bound**: Adjacent values form value _bounds_ which map to the 
  `<color>` associated with the first `<value>` of the boundary.
  The last color value is therefore ignored.
* Type **Key**: The values are integer keys that directly identify 
  the associated color. Should be used for data of type integer.
  Together with a categorical normalisation **CAT**, 
  this color mapping type will also allow for a categorical map legend. 
