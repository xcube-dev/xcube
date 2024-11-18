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

* **Continuous:** Continuous color assignment, where each `<value>` 
  represents a support point of a color gradient.
* **Stepwise:** Stepwise color mapping where values within the range of two
  subsequent `<value>`s are mapped to the same color. A `<color>` gets associated with the 
  first `<value>` of each boundary range, while the last color gets ignored.
* **Categorical:** Values represent unique categories or indexes that are 
  mapped to a color. The data and the `<value>` must be of type integer. 
  If a category does not have a `<value>` in the color mapping, it will be 
  displayed as transparent. Suitable for categorical datasets.


