A user-defined color mapping associates data values or ranges of data values 
with color values. The lines in the text box have the general syntax 
`<value>: <color>`, where `<color>` can be

* a list of RGB values, with values in the range 0 to 255, for example,
  `255,165,0` for the color Orange;
* a hexadecimal RGB value, e.g., `#FFA500`;
* or a valid [HTML color name](https://www.w3schools.com/colors/colors_names.asp)
  such as `Orange`, `BlanchedAlmond` or `MediumSeaGreen`.

Färgvärdet kan kompletteras med ett opacitetsvärde (alpha-värde) i intervallet 0 
till 1, till exempel `110,220,230,0.5` eller `#FFA500,0.8` eller `Blue,0`. 
Hexadecimala värden kan också skrivas med ett alpha-värde, såsom `#FFA500CD`.

Tolkningen av `<value>` beror på den valda färgkartläggningstypen:

* Typ **Node**: Färgkartläggningen är en linjär gradient mellan färgerna med
  relativa avstånd givna av `<value>`, som ger kontrollvärdena eller _noderna_.
* Typ **Bound**: Angränsande värden bildar värdegränser (_bounds_), som kartläggs
  till den `<color>` som är associerad med det första `<value>` av gränsen.
  Det sista färgvärdet ignoreras därför.
* Typ **Key**: Värdena är heltal som direkt identifierar den associerade färgen. 
  Bör användas för data av typen heltal. Tillsammans med en kategorisk normalisering 
  **CAT** möjliggör denna färgkartläggningstyp också en kategorisk kartlegenda.
