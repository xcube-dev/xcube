xcube tile https://s3.eu-central-1.amazonaws.com/esdl-esdc-v2.0.0/esdc-8d-0.083deg-1x2160x4320-2.0.0.zarr ^
  --labels time='2009-01-01/2009-12-30' ^
  --vars analysed_sst,air_temperature_2m ^
  --tile-size 270 ^
  --config .\config-cci-cfs.yml ^
  --style default ^
  --verbose