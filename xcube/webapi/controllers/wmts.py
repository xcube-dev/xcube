import math

from ..context import ServiceContext

# WGS84 ellipsoid semi-major axis
_WGS84_MEAN_EARTH_RADIUS_IN_METERS = 6378137.0
_WGS84_MEAN_EARTH_PERIMETER_IN_METERS = 2.0 * math.pi * _WGS84_MEAN_EARTH_RADIUS_IN_METERS
_WGS84_METERS_PER_DEGREE = _WGS84_MEAN_EARTH_PERIMETER_IN_METERS / 360.0
_STD_PIXEL_SIZE_IN_METERS = 0.28e-3


def get_wmts_capabilities_xml(ctx: ServiceContext, base_url: str):
    service_identification_xml = (
        f"\n"
        f"    <ows:ServiceIdentification>\n"
        f"        <ows:Title>xcube WMTS</ows:Title>\n"
        f"        <ows:Abstract>Web Map Tile Service (WMTS) for xcube-conformant data cubes</ows:Abstract>\n"
        f"        <ows:Keywords>\n"
        f"            <ows:Keyword>tile</ows:Keyword>\n"
        f"            <ows:Keyword>tile matrix set</ows:Keyword>\n"
        f"            <ows:Keyword>map</ows:Keyword>\n"
        f"        </ows:Keywords>\n"
        f"        <ows:ServiceType>OGC WMTS</ows:ServiceType>\n"
        f"        <ows:ServiceTypeVersion>1.0.0</ows:ServiceTypeVersion>\n"
        f"        <ows:Fees>none</ows:Fees>\n"
        f"        <ows:AccessConstraints>none</ows:AccessConstraints>\n"
        f"    </ows:ServiceIdentification>\n"
    )

    service_provider = ctx.config['ServiceProvider']
    service_contact = service_provider['ServiceContact']
    contact_info = service_contact['ContactInfo']
    phone = contact_info['Phone']
    address = contact_info['Address']

    service_provider_xml = (
        f"\n"
        f"    <ows:ServiceProvider>\n"
        f"        <ows:ProviderName>{service_provider['ProviderName']}</ows:ProviderName>\n"
        f"        <ows:ProviderSite xlink:href=\"{service_provider['ProviderSite']}\"/>\n"
        f"        <ows:ServiceContact>\n"
        f"            <ows:IndividualName>{service_contact['IndividualName']}</ows:IndividualName>\n"
        f"            <ows:PositionName>{service_contact['PositionName']}</ows:PositionName>\n"
        f"            <ows:ContactInfo>\n"
        f"                <ows:Phone>\n"
        f"                    <ows:Voice>{phone['Voice']}</ows:Voice>\n"
        f"                    <ows:Facsimile>{phone['Facsimile']}</ows:Facsimile>\n"
        f"                </ows:Phone>\n"
        f"                <ows:Address>\n"
        f"                    <ows:DeliveryPoint>{address['DeliveryPoint']}</ows:DeliveryPoint>\n"
        f"                    <ows:City>{address['City']}</ows:City>\n"
        f"                    <ows:AdministrativeArea>{address['AdministrativeArea']}</ows:AdministrativeArea>\n"
        f"                    <ows:PostalCode>{address['PostalCode']}</ows:PostalCode>\n"
        f"                    <ows:Country>{address['Country']}</ows:Country>\n"
        f"                    <ows:ElectronicMailAddress>{address['ElectronicMailAddress']}"
        f"</ows:ElectronicMailAddress>\n"
        f"                </ows:Address>\n"
        f"            </ows:ContactInfo>\n"
        f"        </ows:ServiceContact>\n"
        f"    </ows:ServiceProvider>\n"
    )

    wmts_kvp_url = ctx.get_service_url(base_url, 'wmts/kvp?')
    wmts_rest_cap_url = ctx.get_service_url(base_url, 'wmts/1.0.0/WMTSCapabilities.xml')
    wmts_rest_tile_url = ctx.get_service_url(base_url, 'wmts/1.0.0/')

    operations_metadata_xml = (
        f"\n"
        f"    <ows:OperationsMetadata>\n"
        f"        <ows:Operation name=\"GetCapabilities\">\n"
        f"            <ows:DCP>\n"
        f"                <ows:HTTP>\n"
        f"                    <ows:Get xlink:href=\"{wmts_kvp_url}\">\n"
        f"                        <ows:Constraint name=\"GetEncoding\">\n"
        f"                            <ows:AllowedValues>\n"
        f"                                <ows:Value>KVP</ows:Value>\n"
        f"                            </ows:AllowedValues>\n"
        f"                        </ows:Constraint>\n"
        f"                    </ows:Get>\n"
        f"                    <ows:Get xlink:href=\"{wmts_rest_cap_url}\">\n"
        f"                        <ows:Constraint name=\"GetEncoding\">\n"
        f"                            <ows:AllowedValues>\n"
        f"                                <ows:Value>REST</ows:Value>\n"
        f"                            </ows:AllowedValues>\n"
        f"                        </ows:Constraint>\n"
        f"                    </ows:Get>\n"
        f"                </ows:HTTP>\n"
        f"            </ows:DCP>\n"
        f"        </ows:Operation>\n"
        f"        <ows:Operation name=\"GetTile\">\n"
        f"            <ows:DCP>\n"
        f"                <ows:HTTP>\n"
        f"                    <ows:Get xlink:href=\"{wmts_kvp_url}\">\n"
        f"                        <ows:Constraint name=\"GetEncoding\">\n"
        f"                            <ows:AllowedValues>\n"
        f"                                <ows:Value>KVP</ows:Value>\n"
        f"                            </ows:AllowedValues>\n"
        f"                        </ows:Constraint>\n"
        f"                    </ows:Get>\n"
        f"                    <ows:Get xlink:href=\"{wmts_rest_tile_url}\">\n"
        f"                        <ows:Constraint name=\"GetEncoding\">\n"
        f"                            <ows:AllowedValues>\n"
        f"                                <ows:Value>REST</ows:Value>\n"
        f"                            </ows:AllowedValues>\n"
        f"                        </ows:Constraint>\n"
        f"                    </ows:Get>\n"
        f"                </ows:HTTP>\n"
        f"            </ows:DCP>\n"
        f"        </ows:Operation>\n"
        f"    </ows:OperationsMetadata>\n"
    )

    dataset_descriptors = ctx.get_dataset_descriptors()
    written_tile_grids = []
    indent = '    '

    layer_base_url = ctx.get_service_url(base_url, 'wmts/1.0.0/tile/%s/%s/{TileMatrix}/{TileRow}/{TileCol}.png')

    dimensions_xml_cache = dict()

    contents_xml_lines = [(0, '<Contents>')]
    for dataset_descriptor in dataset_descriptors:
        ds_name = dataset_descriptor['Identifier']
        ds = ctx.get_dataset(ds_name)
        for var_name in ds.data_vars:
            var = ds[var_name]
            if len(var.shape) <= 2 or var.dims[-1] != 'lon' or var.dims[-2] != 'lat':
                continue

            tile_grid = ctx.get_tile_grid(ds_name)
            tile_grid_written = tile_grid in written_tile_grids
            if tile_grid_written:
                tile_grid_index = written_tile_grids.index(tile_grid)
            else:
                tile_grid_index = len(written_tile_grids)
                written_tile_grids.append(tile_grid)
            tile_grid_id = f"TileGrid_{tile_grid_index}"

            supported_crs = "urn:ogc:def:crs:OGC:1.3:CRS84"
            # supported_crs = "http://www.opengis.net/def/crs/EPSG/9.5.3/4326"

            if tile_grid is not None:
                if not tile_grid_written:
                    tile_size_x, tile_size_y = tile_grid.tile_size
                    lon1, lat1, lon2, lat2 = tile_grid.geo_extent
                    tile_span_y = (lat2 - lat1) / tile_grid.num_level_zero_tiles_y
                    pixel_span = tile_span_y / tile_size_y
                    scale_denominator_0 = pixel_span * _WGS84_METERS_PER_DEGREE / _STD_PIXEL_SIZE_IN_METERS

                    contents_xml_lines.append((2, '<TileMatrixSet>'))
                    contents_xml_lines.append((3, f'<ows:Identifier>{tile_grid_id}</ows:Identifier>'))
                    contents_xml_lines.append((3, f'<ows:SupportedCRS>{supported_crs}</ows:SupportedCRS>'))
                    contents_xml_lines.append((3, '<ows:BoundingBox>'))
                    contents_xml_lines.append((4, f'<ows:LowerCorner>{lon1} {lat1}</ows:LowerCorner>'))
                    contents_xml_lines.append((4, f'<ows:UpperCorner>{lon2} {lat2}</ows:UpperCorner>'))
                    contents_xml_lines.append((3, '</ows:BoundingBox>'))

                    for level in range(tile_grid.num_levels):
                        factor = 2 ** level
                        num_tiles_x = tile_grid.num_level_zero_tiles_x * factor
                        num_tiles_y = tile_grid.num_level_zero_tiles_y * factor
                        scale_denominator = scale_denominator_0 / factor
                        contents_xml_lines.append((3, '<TileMatrix>'))
                        contents_xml_lines.append((4, f'<ows:Identifier>{level}</ows:Identifier>'))
                        contents_xml_lines.append((4, f'<ScaleDenominator>{scale_denominator}</ScaleDenominator>'))
                        contents_xml_lines.append((4, f'<TopLeftCorner>{lon1} {lat2}</TopLeftCorner>'))
                        contents_xml_lines.append((4, f'<TileWidth>{tile_size_x}</TileWidth>'))
                        contents_xml_lines.append((4, f'<TileHeight>{tile_size_y}</TileHeight>'))
                        contents_xml_lines.append((4, f'<MatrixWidth>{num_tiles_x}</MatrixWidth>'))
                        contents_xml_lines.append((4, f'<MatrixHeight>{num_tiles_y}</MatrixHeight>'))
                        contents_xml_lines.append((3, '</TileMatrix>'))

                    contents_xml_lines.append((2, '</TileMatrixSet>'))

                var_title = ds_name + "/" + var.attrs.get('title', var.attrs.get('long_name', var_name))
                var_abstract = var.attrs.get('comment', '')

                layer_tile_url = layer_base_url % (ds_name, var_name)
                contents_xml_lines.append((2, '<Layer>'))
                contents_xml_lines.append((3, f'<ows:Identifier>{ds_name}.{var_name}</ows:Identifier>'))
                contents_xml_lines.append((3, f'<ows:Title>{var_title}</ows:Title>'))
                contents_xml_lines.append((3, f'<ows:Abstract>{var_abstract}</ows:Abstract>'))
                contents_xml_lines.append((3, '<ows:WGS84BoundingBox>'))
                contents_xml_lines.append((4, f'<ows:LowerCorner>{lon1} {lat1}</ows:LowerCorner>'))
                contents_xml_lines.append((4, f'<ows:UpperCorner>{lon2} {lat2}</ows:UpperCorner>'))
                contents_xml_lines.append((3, '</ows:WGS84BoundingBox>'))
                contents_xml_lines.append(
                    (3, '<Style isDefault="true"><ows:Identifier>Default</ows:Identifier></Style>'))
                contents_xml_lines.append((3, '<Format>image/png</Format>'))
                contents_xml_lines.append(
                    (3, f'<TileMatrixSetLink><TileMatrixSet>{tile_grid_id}</TileMatrixSet></TileMatrixSetLink>'))
                contents_xml_lines.append(
                    (3, f'<ResourceURL format="image/png" resourceType="tile" template="{layer_tile_url}"/>'))

                non_spatial_dims = var.dims[0:-2]
                for dim_name in non_spatial_dims:
                    if dim_name not in ds.coords:
                        continue
                    dimension_xml_key = f'{ds_name}.{dim_name}'
                    if dimension_xml_key in dimensions_xml_cache:
                        dimensions_xml_lines = dimensions_xml_cache[dimension_xml_key]
                    else:
                        coord_var = ds.coords[dim_name]
                        if len(coord_var.shape) != 1:
                            # strange case
                            continue
                        coord_bnds_var_name = coord_var.attrs.get('bounds', dim_name + '_bnds')
                        coord_bnds_var = ds.coords[coord_bnds_var_name] if coord_bnds_var_name in ds else None
                        if coord_bnds_var is not None:
                            if len(coord_bnds_var.shape) != 2 \
                                    or coord_bnds_var.shape[0] != coord_bnds_var.shape[0] \
                                    or coord_bnds_var.shape[1] != 2:
                                # strange case
                                coord_bnds_var = None
                        var_title = coord_var.attrs.get('long_name', dim_name)
                        units = 'ISO8601' if dim_name == 'time' else coord_var.attrs.get('units', '')
                        default = 'current' if dim_name == 'time' else '0'
                        current = 'true' if dim_name == 'time' else 'false'
                        dimensions_xml_lines = [(3, '<Dimension>'),
                                                (4, f'<ows:Identifier>{dim_name}</ows:Identifier>'),
                                                (4, f'<ows:Title>{var_title}</ows:Title>'),
                                                (4, f'<ows:UOM>{units}</ows:UOM>'),
                                                (4, f'<Default>{default}</Default>'),
                                                (4, f'<Current>{current}</Current>')]
                        if coord_bnds_var is not None:
                            coord_bnds_var_values = coord_bnds_var.values
                            for i in range(len(coord_var)):
                                value1 = coord_bnds_var_values[i, 0]
                                value2 = coord_bnds_var_values[i, 1]
                                dimensions_xml_lines.append((4, f'<Value>{value1}/{value2}</Value>'))
                        else:
                            coord_var_values = coord_var.values
                            for i in range(len(coord_var)):
                                value = coord_var_values[i]
                                dimensions_xml_lines.append((4, f'<Value>{value}</Value>'))
                        dimensions_xml_lines.append((3, '</Dimension>'))
                        dimensions_xml_cache[dimension_xml_key] = dimensions_xml_lines

                    contents_xml_lines.extend(dimensions_xml_lines)
                contents_xml_lines.append((2, '</Layer>'))

    contents_xml_lines.append((1, '</Contents>'))

    contents_xml = '\n'.join(['%s%s' % (n * indent, xml) for n, xml in contents_xml_lines])

    themes_xml_lines = [(0, '<Themes>')]
    for dataset_descriptor in dataset_descriptors:
        ds_name = dataset_descriptor.get('Identifier')
        ds = ctx.get_dataset(ds_name)
        ds_title = dataset_descriptor.get('Title', ds.attrs.get('title', f'{ds_name} xcube dataset'))
        ds_abstract = ds.attrs.get('comment', '')
        themes_xml_lines.append((2, '<Theme>'))
        themes_xml_lines.append((3, f'<ows:Title>{ds_title}</ows:Title>'))
        themes_xml_lines.append((3, f'<ows:Abstract>{ds_abstract}</ows:Abstract>'))
        themes_xml_lines.append((3, f'<ows:Identifier>{ds_name}</ows:Identifier>'))
        for var_name in ds.data_vars:
            var = ds[var_name]
            var_title = var.attrs.get('title', var.attrs.get('long_name', var_name))
            themes_xml_lines.append((3, '<Theme>'))
            themes_xml_lines.append((4, f'<ows:Title>{var_title}</ows:Title>'))
            themes_xml_lines.append((4, f'<ows:Identifier>{ds_name}.{var_name}</ows:Identifier>'))
            themes_xml_lines.append((4, f'<LayerRef>{ds_name}.{var_name}</LayerRef>'))
            themes_xml_lines.append((3, '</Theme>'))
        themes_xml_lines.append((2, '</Theme>'))
    themes_xml_lines.append((1, '</Themes>'))
    themes_xml = '\n'.join(['%s%s' % (n * indent, xml) for n, xml in themes_xml_lines])

    get_capablities_rest_url = ctx.get_service_url(base_url, 'wmts/1.0.0/WMTSCapabilities.xml')
    service_metadata_url_xml = f'<ServiceMetadataURL xlink:href="{get_capablities_rest_url}"/>'

    return (
        f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<Capabilities xmlns=\"http://www.opengis.net/wmts/1.0\"\n"
        f"          xmlns:ows=\"http://www.opengis.net/ows/1.1\"\n"
        f"          xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n"
        f"          xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
        f"          xsi:schemaLocation=\"http://www.opengis.net/wmts/1.0"
        f" http://schemas.opengis.net/wmts/1.0.0/wmtsGetCapabilities_response.xsd\"\n"
        f"          version=\"1.0.0\">\n"
        f"    {service_identification_xml}\n"
        f"    {service_provider_xml}\n"
        f"    {operations_metadata_xml}\n"
        f"    {contents_xml}\n"
        f"    {themes_xml}\n"
        f"    {service_metadata_url_xml}\n"
        f"</Capabilities>\n"
    )
