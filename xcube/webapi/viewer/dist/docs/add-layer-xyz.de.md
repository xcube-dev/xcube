Der Name XYZ bezieht sich auf die URLs, die von Diensten verwendet werden, die 
[Tiled Web Maps](https://en.wikipedia.org/wiki/Tiled_web_map) bereitstellen, 
oft auch als [OpenStreetMap (OSM)](https://en.wikipedia.org/wiki/OpenStreetMap) 
Standard oder _Slippy Maps_ bezeichnet. Die URLs werden auch häufig von Kartenservern
verwendet, die den [Tile Map Service (TMS)](https://en.wikipedia.org/wiki/Tile_Map_Service) 
Standard implementieren. Die URLs für eine solche Karte enthalten die x- und y-Koordinaten 
einer Bildkachel und einen optionalen Zoomlevel z. Zum Beispiel,
`https://a.tile.osm.org/{z}/{x}/{y}.png`. 

**XYZ Layer URL**: Die URL des Layers. Diese muss die folgenden Muster enthalten 
`{x}`, `{y}`, und optional `{z}`. `{-y}` kann verwendet werden, um
eine gespiegelte y-Achse anzugeben.

**Layer Titel**: Der beschreibende Titel für den Layer.

**Layer Attribution**: Optionale Attributionsinformationen für den Layer.