
def compute_dataset(ds, period='1W'):
    return ds.resample(time=period).mean(dim='time')
