Namnet _XYZ_ hänvisar till de webbadresser som används av tjänster som tillhandahåller 
[Tiled Web Maps](https://en.wikipedia.org/wiki/Tiled_web_map) ofta också 
refereras till som [OpenStreetMap (OSM)](https://en.wikipedia.org/wiki/OpenStreetMap)
standard eller _Slippy Maps_. URL:erna används också ofta av kartservrar som
implementerar [Tile Map Service (TMS)](https://en.wikipedia.org/wiki/Tile_Map_Service)
standard. URL-adresserna för en sådan karta innehåller en bildkakels x- och y-koordinater 
och en valfri zoomnivå z. Till exempel, 
`https://a.tile.osm.org/{z}/{x}/{y}.png`. 

**XYZ lager URL**: URL-adressen till lagern. Den måste innehålla mönstren 
`{x}`, `{y}` och eventuellt `{z}`. Observera att `{-y}` kan användas för att
för att ange en vänd y-axel.

**Lagertitel**: Den beskrivande titeln för lager.

**Lagerattribution**: Optionell attributionsinformation för lagret.