import os

import pandas as pd
import s3fs

from xcube.core import read_cube


def show_remote_cubes(bucket, endpoint_url, region_name):

    s3_client_kwargs = {}
    s3_client_kwargs['endpoint_url'] = endpoint_url
    s3_client_kwargs['region_name'] = region_name
    obs_file_system = s3fs.S3FileSystem(anon=True, client_kwargs=s3_client_kwargs)

    cube_names = []
    df = pd.DataFrame(
        columns=['cube_name', 'chunksize (time, lat, lon)', 'var_nums', 'variables', 'start_date', 'end_date'])

    for filepath in sorted(obs_file_system.ls(bucket)):
        if filepath.endswith('.zarr'):
            # ds = read_cube(f'{endpoint_url}/{filepath}', consolidated=True)
            ds = read_cube(f'{endpoint_url}/{filepath}')
            var_list = list(ds.data_vars)
            cube_names.append(filepath)
            filename = filepath.split('/')[1]
            sd = pd.to_datetime(str(ds.time.values[0]))
            start_date = sd.strftime('%Y-%m-%d')
            ed = pd.to_datetime(str(ds.time.values[-1]))
            end_date = ed.strftime('%Y-%m-%d')
            chunksize = ds[var_list[0]].data.chunksize
            df = df.append({'cube_name': filename,
                            'chunks (time, lat, lon)': chunksize,
                            'num_vars': len(var_list),
                            'variables': (str(var_list)).replace('[', '').replace(']', '').replace("'", ""),
                            'start_date': start_date,
                            'end_date': end_date}, ignore_index=True)
    return df
