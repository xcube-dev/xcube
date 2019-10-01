function newTileXYZSourceFromJsonOptions(jsonOptions) {
    return new ol.source.XYZ({
        url: jsonOptions.url,
        projection: ol.proj.get(jsonOptions.projection),
        minZoom: jsonOptions.minZoom,
        maxZoom: jsonOptions.maxZoom,
        tileGrid: new ol.tilegrid.TileGrid(jsonOptions.tileGrid)
    });
}

function fetchTileXYZSource(url) {
    return fetch(url)
            .then(response => response.json())
            .then(options => {
                return newTileXYZSourceFromJsonOptions(options);
            });
}

function fetchTileLayer(url) {
    return fetchTileXYZSource(url).then(source => new ol.layer.Tile({source: source}));
}