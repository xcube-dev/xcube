xcube-genl2c ^
   -t snap-olci-highroc-l2 ^
   -f zarr ^
   -a ^
   -s 2000,1000 ^
   -r 0,50,5,52.5 ^
   -v conc_chl,conc_tsm,kd489,c2rcc_flags,quality_flags ^
   -m D:\Projects\xcube\examples\dcs4cop-metadata.yml ^
   -d D:\EOData\DCS4COP ^
   -n OLCI-SNS-RAW-CUBE-2 ^
   D:\EOData\HIGHROC\0001_SNS\OLCI\**\*.nc