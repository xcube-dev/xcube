# How do I ...

## Data Access

| How do I ...                        | Use API                                                                        |
|-------------------------------------|--------------------------------------------------------------------------------|
| get a data store instance           | [new_data_store(store_id, ...)](api.html#xcube.core.store.new_data_store)      |
| get data store for local filesystem | [new_data_store("file", ...)](api.html#xcube.core.store.new_data_store)        |
| get data store for S3 filesystem    | [new_data_store("s3", ...)](api.html#xcube.core.store.new_data_store)          |
| list the datasets in a data store   | [store.list_data_ids()](api.html#xcube.core.store.DataStore.list_data_ids)     |
| open a dataset from a data store    | [store.open_data(data_id, ...)](api.html#xcube.core.store.DataStore.open_data) |

## Resampling

| How do I ...                                | Use API                                                                 |
|---------------------------------------------|-------------------------------------------------------------------------|
| spatially resample a datacube               | [resample_in_space()](api.html#xcube.core.resampling.resample_in_space) |
| temporarily resample a datacube             | [resample_in_time()](api.html#xcube.core.resampling.resample_in_time)   |
| rectify a dataset in satellite projection   | [rectify_dataset()](api.html#xcube.core.resampling.rectify_dataset)     |

