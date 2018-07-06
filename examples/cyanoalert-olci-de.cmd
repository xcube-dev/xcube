xcube-genl2c ^
   -t snap-olci-cyanoalert-l2 ^
   -f zarr ^
   -a ^
   -s 1000,410 ^
   -r 9.4,53.5,11.4,54.72 ^
   -v "chl,conc_chl,conc_tsm,iop_adg,iop_agelb,kd_z90max,tur_nechad_665,tur_nechad_865" ^
   -m D:\Projects\xcube\examples\cyanoalert-olci-metadata.yml ^
   -d D:\EOData\CyanoAlert ^
   -n OLCI-DE-L2C-CUBE ^
   D:\EOData\CyanoAlert\*.nc
