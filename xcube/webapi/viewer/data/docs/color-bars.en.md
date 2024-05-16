A color bar comprises mappings from a sample value to a color value. 
Each line in the text box is such a mapping. 
The general syntax for the lines is `<value>: <color>`, where
`<value>` can be any number. In the end, the values are normalized 
to the range 0 to 1 and the list of color mappings is sorted
by sample value. `<color>` can be 

* a list of RGB values, with values in the range 0 to 255, for example,
  `255,165,0` for the color Orange;
* a hexadecimal RGB value, e.g., `#FFA500`;
* or a valid [HTML color name](https://www.w3schools.com/colors/colors_names.asp)
  such as `Orange`, `BlanchedAlmond` or `MediumSeaGreen`.  

The color value may be suffixed by a opaqueness (alpha) value in the range
0 to 1, for example `110,220,230,0.5` or `#FFA500,0.8` or `Blue,0`. 
Hexadecimal values can also be written including an alpha value, 
such as `#FFA500CD`.
