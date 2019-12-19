import os

import pandas as pd
import s3fs

from xcube.core.dsio import open_cube


def show_remote_cubes(bucket, endpoint_url, region_name):

    s3_client_kwargs = {}
    s3_client_kwargs['endpoint_url'] = endpoint_url
    s3_client_kwargs['region_name'] = region_name
    obs_file_system = s3fs.S3FileSystem(anon=True, client_kwargs=s3_client_kwargs)

    cube_names = []
    df = pd.DataFrame(
        columns=['cube_name', 'chunks (time, lat, lon)', 'number_of_variables', 'variables',
                 'start_date', 'end_date', 'spatial_coverage (lon_min, lat_min, lon_max, lat_max)'])

    for filepath in sorted(obs_file_system.ls(bucket)):
        if filepath.endswith('.zarr'):
            # ds = read_cube(f'{endpoint_url}/{filepath}', consolidated=True)
            with open_cube(f'{endpoint_url}/{filepath}') as ds:
                var_list = list(ds.data_vars)
                cube_names.append(filepath)
                filename = filepath.split('/')[1]
                sd = pd.to_datetime(str(ds.time.values[0]))
                start_date = sd.strftime('%Y-%m-%d')
                ed = pd.to_datetime(str(ds.time.values[-1]))
                end_date = ed.strftime('%Y-%m-%d')
                chunksize = ds[var_list[0]].data.chunksize
                try:
                    spat_cov = (ds.attrs['geospatial_lon_min'], ds.attrs['geospatial_lat_min'],
                                ds.attrs['geospatial_lon_max'], ds.attrs['geospatial_lat_max'])
                except KeyError:
                    spat_cov = None
                df = df.append({'cube_name': filename,
                                'chunks (time, lat, lon)': chunksize,
                                'number_of_variables': len(var_list),
                                'variables': (str(var_list)).replace('[', '').replace(']', '').replace("'", ""),
                                'start_date': start_date,
                                'end_date': end_date,
                                'spatial_coverage (lon_min, lat_min, lon_max, lat_max)': str(spat_cov)}, ignore_index=True)
    return df
