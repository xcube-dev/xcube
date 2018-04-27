import sys
from typing import List, Optional

import xarray as xr

from xcube.reproject import reproject_to_wgs84
from xcube.snap.mask import mask_dataset


def main(args: Optional[List[str]] = None):
    """
    Masks and reprojects a HIGHROC L2 OLCI NetCDF4 product.

    Tested with HIGHROC/0001_SNS/OLCI/2017/04/O_L2_0001_SNS_2017104102450_v1.0.nc

    :param args: <MY-HIGHROC-L2-OLCI-FILE.nc> [<MY-OUTPUT.nc>]
    :return:
    """
    args = args or sys.argv[1:]
    input_path = args[0]
    output_path = args[1] if len(args) == 2 else 'highroc-test-out.nc'

    dataset = xr.open_dataset(input_path, decode_cf=True, decode_coords=True)

    masked_dataset, mask_sets = mask_dataset(dataset,
                                             expr_pattern='({expr}) AND !quality_flags.land',
                                             errors='raise')

    for _, mask_set in mask_sets.items():
        print('MaskSet: %s' % mask_set)

    dst_width = 1024
    dst_height = 512
    new_dataset = reproject_to_wgs84(masked_dataset, dst_width, dst_height, gcp_i_step=50)

    _rm(output_path)
    new_dataset.to_netcdf(output_path)

    # from matplotlib import pyplot as plt
    # for var_name in new_dataset.variables:
    #     var = new_dataset[var_name]
    #     if var.dims == ('lat', 'lon'):
    #         var.plot()
    #         plt.show()


def _rm(path):
    import os
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except:
            pass


if __name__ == '__main__':
    main()
