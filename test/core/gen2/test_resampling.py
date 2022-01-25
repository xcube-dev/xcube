import os
import unittest

from xcube.core.gen2 import CubeConfig
from xcube.core.gen2 import CubeGenerator
from xcube.core.gen2 import CubeGeneratorRequest
from xcube.core.gen2 import CubeGeneratorResult
from xcube.core.gen2 import CubeInfoResult
from xcube.core.gen2 import InputConfig
from xcube.core.gen2 import OutputConfig
from xcube.core.store import DataStoreConfig
from xcube.core.store import DataStorePool

S3_KEY_KEY = "AWS_ACCESS_KEY_ID"
S3_SECRET_KEY = "AWS_SECRET_ACCESS_KEY"

S3_KEY = os.environ.get(S3_KEY_KEY)
S3_SECRET = os.environ.get(S3_SECRET_KEY)

TEST_ENABLED = True


@unittest.skipUnless(all([TEST_ENABLED, S3_KEY, S3_SECRET]),
                     f"AWS S3 environment variables {S3_KEY_KEY} "
                     f"and {S3_SECRET_KEY} must be set")
class ResamplingTest(unittest.TestCase):
    """
    Investigate resampling performed by xcube Generator.
    """

    def test_resampling(self):
        stores_config = DataStorePool(dict(
            cop_services_data=DataStoreConfig(
                store_id="s3",
                store_params=dict(
                    root='cop-services',
                    storage_options=dict(
                        anon=False,
                        key=S3_KEY,
                        secret=S3_SECRET
                    )
                )
            )
        ))

        med_sea_bbox = (-5.993576, 30.005064, 36.493576, 45.994938)
        # north_sea_bbox = (-45.994793, 20.005207, 12.994794, 65.99479)
        # bs_bbox = (26.507038, 40.005062, 41.99296, 47.994938)
        # bal_bbox = (9.258861, 53.255493, 30.241138, 65.844505)

        input_config = InputConfig(
            store_id="@cop_services_data",
            data_id='OCEANCOLOUR_MED_CHL_L4_NRT_OBSERVATIONS_009_041.zarr',
        )

        cube_config = CubeConfig(
            variable_names=["CHL"],
            tile_size=(512, 512),
            spatial_res=0.025000000002,
            bbox=med_sea_bbox,
            time_range=("2019-01-01", "2019-03-15"),
            time_period="1D",
            metadata=dict(title='CMEMS Med Sea Chl')
        )

        output_config = OutputConfig(
            store_id="file",
            store_params=dict(root='.'),
            replace=True,
            data_id="CMEMS_OC_Med_4.zarr",
        )

        request = CubeGeneratorRequest(
            input_config=input_config,
            output_config=output_config,
            cube_config=cube_config
        )

        generator = CubeGenerator.new(
            stores_config=stores_config,
            verbosity=3
        )

        cube_info_result = generator.get_cube_info(request)
        self.assertIsInstance(cube_info_result, CubeInfoResult)

        cube_result = generator.generate_cube(request)
        self.assertIsInstance(cube_result, CubeGeneratorResult)
