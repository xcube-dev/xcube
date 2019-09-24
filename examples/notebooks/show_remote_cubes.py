import os

import pandas as pd
import s3fs

from xcube.api import read_cube


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
            ds = read_cube(os.path.join('http://obs.eu-de.otc.t-systems.com', filepath))
            var_list = []
            for v in ds.variables:
                if v not in ds.dims and 'bnds' not in v:
                    var_list.append(v)
            cube_names.append(filepath)
            filename = filepath.split('/')[1]
            sd = pd.to_datetime(str(ds.time.values[0]))
            start_date = sd.strftime('%Y-%m-%d')
            ed = pd.to_datetime(str(ds.time.values[-1]))
            end_date = ed.strftime('%Y-%m-%d')
            chunksize = ds[var_list[0]].data.chunksize
            df = df.append({'cube_name': filename, 'chunksize (time, lat, lon)': chunksize,
                            'var_nums': len(var_list),
                            'variables': (str(var_list)).replace('[', '').replace(']', '').replace("'", ""),
                            'start_date': start_date, 'end_date': end_date}, ignore_index=True)
    return df
