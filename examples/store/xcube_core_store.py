import os
import os.path

import xarray as xr


def get_data_store_ids():
    return list(REGISTRY.keys())


def get_data_store(data_store_id):
    return REGISTRY.get(data_store_id)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class LocalDataStore:
    OPEN_FORMATS = {
        '.zarr': 'zarr',
        '.nc': 'netcdf',
        '.hdf': 'netcdf',
        '.h5': 'netcdf',
        '.geojson': 'geojson',
    }

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or '.'

    @classmethod
    def get_service_params_schema(cls):
        return dict(
            type='object',
            properties=dict(
                base_dir=dict(type='string', min_length=1),
            ),
            additional_properties=False,
            required=[],
        )

    def get_dataset_ids(self):
        dataset_ids = []
        for name in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, name)
            if path.endswith('.zarr') and os.path.isdir(path):
                dataset_ids.append(name)
            elif path.endswith('.nc') and os.path.isfile(path):
                dataset_ids.append(name)
        return dataset_ids

    def _open_dataset(self, dataset_id: str):
        _, ext = os.path.splitext(dataset_id)
        format_name = self.OPEN_FORMATS.get(ext)
        if format_name == 'zarr' or format_name == 'zarr:zip':
            return xr.open_zarr(os.path.join(self.base_dir, dataset_id))
        else:
            return xr.open_dataset(os.path.join(self.base_dir, dataset_id))
        # TODO: also mock reading geojson + shapefiles

    def get_dataset_descriptor(self, dataset_id: str):
        ds = self._open_dataset(dataset_id)

        _, ext = os.path.splitext(dataset_id)
        format_name = self.OPEN_FORMATS.get(ext)

        if format_name == 'zarr':
            zarr_params = _get_common_gridded_constraint_params(list(ds.data_vars.keys()), ['WGS84'])
            zarr_params.update(group=dict(type='string'),
                               chunks=dict(type='object'),
                               decode_cf=dict(type='boolean'),
                               consolidated=dict(type='boolean'))
            format_desc = dict(name='zarr',
                               params_schema=_get_params_schema(zarr_params, required=[]))
        else:
            netcdf_params = _get_common_gridded_constraint_params(list(ds.data_vars.keys()), ['WGS84'])
            netcdf_params.update(group=dict(type='string'),
                                 chunks=dict(type='object'),
                                 engine=dict(type='string'),
                                 cache=dict(type='boolean'),
                                 decode_cf=dict(type='boolean'),
                                 consolidated=dict(type='boolean'))
            format_desc = dict(name='netcdf',
                               params_schema=_get_params_schema(netcdf_params, required=[]))

        return dict(
            id=dataset_id,
            description=ds.attrs.get('description', ds.attrs.get('title')),
            variable_names=list(ds.data_vars.keys()),
            time_range=(str(ds.time.values[0]), str(ds.time.values[-1]))
            if 'time' in ds and ds.time.ndim == 1 else None,
            bbox=tuple(map(float, (ds.lon[0], ds.lat[0], ds.lon[-1], ds.lat[-1])))
            if 'lon' in ds and ds.lon.ndim == 1 and 'lat' in ds and ds.lat.ndim == 1 else None,
            open_formats=[format_desc],
        )

    def get_open_dataset_format_names(self, dataset_id: str):
        ds_desc = self.get_dataset_descriptor(dataset_id)
        return [format_desc['name'] for format_desc in ds_desc['open_formats']]

    def get_open_dataset_params_schema(self, dataset_id, format_name=None):
        ds_desc = self.get_dataset_descriptor(dataset_id)
        format_names = self.get_open_dataset_format_names(dataset_id)
        if format_name is None:
            format_name = format_names[0]
        else:
            if format_name not in format_names:
                raise ValueError(f'format {format_name} not supported')
        for format_desc in ds_desc['open_formats']:
            if format_desc['name'] == format_name:
                return format_desc['params_schema']
        return None

    def open_dataset(self, dataset_id, format_name=None, **open_params):
        if format_name is None:
            _, ext = os.path.splitext(dataset_id)
            format_name = self.OPEN_FORMATS.get(ext)
        if format_name == 'zarr':
            return xr.open_zarr(os.path.join(self.base_dir, dataset_id), **open_params)
        else:
            return xr.open_dataset(os.path.join(self.base_dir, dataset_id), **open_params)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class CCIDataStore:
    SST_VARS = [
        'analysed_sst',
        'analysis_error',
        'sea_ice_fraction_error',
        'sea_ice_fraction',
        'mask'
    ]

    def __init__(self, api_url=None):
        pass

    @classmethod
    def get_service_params_schema(cls):
        return dict(
            type='object',
            properties=dict(
                api_url=dict(type='string', default='http://opensearch-test.ceda.ac.uk/opensearch/request',
                             min_length=1, format='uri'),
            ),
            additional_properties=False,
            required=[],
        )

    def get_dataset_ids(self):
        with open('./cci-dataset-ids.txt') as fp:
            return list(fp.readlines())

    def get_dataset_descriptor(self, dataset_id):
        return dict(
            id='esacci2.SST.day.L4.SSTdepth.multi-sensor.multi-platform.OSTIA.1-1.r1',
            description='ESA Sea Surface Temperature Climate Change Initiative '
                        '(ESA SST CCI): Analysis long term product version 1.1',
            variable_names=self.SST_VARS,
            bbox=[-180, -90, 180, 90],
            spatial_res=1 / 20,
            time_range=['1991-08-31T23:00:00', '2010-12-31T00:00:00'],
            time_period='1D',
            open_formats=[dict(name=f, params_schema=self.get_open_dataset_params_schema(dataset_id, f)) for f in
                          self.get_open_dataset_format_names(dataset_id)],
        )

    def get_open_dataset_format_names(self, dataset_id):
        return ['zarr:cciodp']

    def get_open_dataset_params_schema(self, dataset_id, format_name=None):
        params = _get_common_gridded_constraint_params(self.SST_VARS, ['WGS84'])
        params.update(collection_id=dict(type=['null', 'string'], min_length=10, default=None)),
        return _get_params_schema(params, defaults=None, required=[])

    def open_dataset(self, dataset_id, format_name=None, **open_params):
        # return xr.tutorial.open_dataset('ROMS_example.nc', chunks={'ocean_time': 1})
        return xr.tutorial.open_dataset('air_temperature')


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class SHDataStore:
    S2L2A_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

    def __init__(self, api_url=None, oauth2_url=None, client_id=None, client_secret=None, instance_id=None):
        pass

    @classmethod
    def get_service_params_schema(cls):
        return dict(
            type='object',
            properties=dict(
                api_url=dict(type='string', default='https://services.sentinel-hub.com/api/v1', min_length=1,
                             format='uri'),
                oauth2_url=dict(type='string', default='https://services.sentinel-hub.com/oauth', min_length=1,
                                format='uri'),
                client_id=dict(type='string', min_length=1),
                client_secret=dict(type='string', min_length=1),
                instance_id=dict(type='string', min_length=1),
            ),
            additional_properties=False,
            required=['client_id', 'client_secret'],
        )

    def get_dataset_ids(self):
        return ['S1GRD', 'S2L1A', 'S2L1B', 'S2L2A', 'S3L1B']

    def get_dataset_descriptor(self, dataset_id):
        return dict(
            id=dataset_id,
            description='Sentinel-2 L2A Product',
            open_formats=[dict(name=f, params_schema=self.get_open_dataset_params_schema(dataset_id, f)) for f in
                          self.get_open_dataset_format_names(dataset_id)],
            variable_names=self.S2L2A_BANDS,
        )

    def get_open_dataset_format_names(self, dataset_id):
        return ['zarr:sentinelhub']

    def get_open_dataset_params_schema(self, dataset_id, format_name=None):
        params = _get_common_gridded_constraint_params(self.S2L2A_BANDS, ['WGS84'])
        params.update(collection_id=dict(type=['null', 'string'], min_length=10, default=None)),
        return _get_params_schema(params,
                                  defaults=dict(
                                      variable_names=[],
                                      crs='WGS84',
                                      time_period='1D',
                                      collection_id=None,
                                  ))

    def open_dataset(self, dataset_id, format_name=None, **open_params):
        # return xr.tutorial.open_dataset('ROMS_example.nc', chunks={'ocean_time': 1})
        return xr.tutorial.open_dataset('air_temperature')


REGISTRY = {
    'sentinelhub': SHDataStore,
    'cciodp': CCIDataStore,
    'c3s': None,
    'geodb': None,
    's3': None,
    'mem': None,
    'local': LocalDataStore,
}


def _get_params_schema(params, defaults=None, required=None):
    if defaults is not None:
        for k, v in params.items():
            if 'default' not in v and k in defaults:
                v['default'] = defaults[k]
    if required is None:
        required = [k for k, v in params.items() if 'default' not in v]
    return dict(
        type='object',
        properties=params,
        additional_properties=False,
        required=required,
    )


def _get_common_gridded_constraint_params(variable_names, crs_names):
    return dict(
        variable_names=dict(type=['array', 'null'],
                            min_items=1, unique_items=True,
                            items=dict(type='string',
                                       enum=variable_names)),
        crs=dict(type='string',
                 enum=crs_names,
                 default=crs_names[0]),
        bbox=dict(type='array',
                  items=[dict(type='number'),
                         dict(type='number'),
                         dict(type='number'),
                         dict(type='number')]),
        spatial_res=dict(type='number',
                         exclusive_minimum=0),
        time_range=dict(type='array',
                        items=[dict(type='str', format='date-time'),
                               dict(type='str', format='date-time')]),
        time_period=dict(type=['string', 'null'],
                         exclusive_minimum=0),
    )
