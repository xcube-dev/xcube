The name _XYZ_ refers to the URLs used by services that provide 
[Tiled Web Maps](https://en.wikipedia.org/wiki/Tiled_web_map) often also 
referred to as [OpenStreetMap (OSM)](https://en.wikipedia.org/wiki/OpenStreetMap)
standard or _Slippy Maps_. The URLs are also commonly used by map server that
implement the [Tile Map Service (TMS)](https://en.wikipedia.org/wiki/Tile_Map_Service)
standard. The URLs for such a map contain an image tile's x- and y-coordinates 
and an optional zoom level z. For example, 
`https://a.tile.osm.org/{z}/{x}/{y}.png`. 

**XYZ Layer URL**: The URL of the layer. It must contain the patterns 
`{x}`, `{y}`, and optionally `{z}`. Note that `{-y}` may be used to
indicate a flipped y-axis.

**Layer Title**: The descriptive title for the layer.

**Layer Attribution**: Optional attribution information for the layer.
