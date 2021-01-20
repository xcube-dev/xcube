import pandas as pd
import s3fs

from xcube.core.dsio import open_cube


def show_remote_cubes(bucket, endpoint_url, region_name='eu-central-1'):
    s3_client_kwargs = {}
    s3_client_kwargs['endpoint_url'] = endpoint_url
    s3_client_kwargs['region_name'] = region_name
    obs_file_system = s3fs.S3FileSystem(anon=True, client_kwargs=s3_client_kwargs)

    cube_names = []
    df = pd.DataFrame(
        columns=['cube_name', 'chunks', 'number_of_variables', 'variables',
                 'start_date', 'end_date', 'spatial_coverage'])

    for filepath in sorted(obs_file_system.ls(bucket)):
        if filepath.endswith('.zarr'):
            with open_cube(f'{endpoint_url}/{filepath}', s3_kwargs=dict(anon=True)) as ds:
                var_list = list(ds.data_vars)
                cube_names.append(filepath)
                filename = filepath.split('/')[1]
                sd = pd.to_datetime(str(ds.time.values[0]))
                start_date = sd.strftime('%Y-%m-%d')
                ed = pd.to_datetime(str(ds.time.values[-1]))
                end_date = ed.strftime('%Y-%m-%d')
                chunksize = []
                for idx, dim in enumerate(ds[var_list[0]].dims):
                    chunksize.append(f"{dim}: {ds[var_list[0]].data.chunksize[idx]}")
                try:
                    spat_cov = ([
                        f"lon_min: {ds.attrs['geospatial_lon_min']}",
                        f"lat_min: {ds.attrs['geospatial_lat_min']}",
                        f"lon_max: {ds.attrs['geospatial_lon_max']}",
                        f"lat_max: {ds.attrs['geospatial_lat_max']}"])
                    spat_cov = ', '.join(spat_cov)
                except KeyError:
                    spat_cov = 'None'
                df = df.append({'cube_name': filename,
                                'chunks': ', '.join(chunksize),
                                'number_of_variables': len(var_list),
                                'variables': ', '.join(var_list),
                                'start_date': start_date,
                                'end_date': end_date,
                                'spatial_coverage': spat_cov},
                               ignore_index=True)
    # Make the variables column wide enough:
    df.style.set_properties(subset=['variables'], width='300px')
    return df


bucket = 'xcube-examples'
endpoint_url = 'https://s3.eu-central-1.amazonaws.com'

overview_cubes_table = show_remote_cubes(bucket, endpoint_url)
