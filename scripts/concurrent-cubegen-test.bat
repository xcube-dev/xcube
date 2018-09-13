@echo on
set DATA_DIR="X:\temp\concurrent-cubegen-test"
set CUBE_DIR="%DATA_DIR%\cube.zarr"
set FILE_POSTFIX=120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_LT-v02.0-fv01.0.nc
call activate xcube-dev
rem start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000725%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000726%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000727%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000728%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000729%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000730%FILE_POSTFIX%"
start /B /I python concurrent-cubegen-test.py %CUBE_DIR% "%DATA_DIR%\20000731%FILE_POSTFIX%"
