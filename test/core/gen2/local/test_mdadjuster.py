import unittest

import pyproj

from xcube.core.gen2 import CubeConfig
from xcube.core.gen2.local.mdadjuster import CubeMetadataAdjuster
from xcube.core.gridmapping import CRS_CRS84
from xcube.core.gridmapping import GridMapping
from xcube.core.new import new_cube


class CubeMetadataAdjusterTest(unittest.TestCase):

    def test_metadata_adjusted_geo_crs(self):
        (x1, x2), (y1, y2) = (53, 54), (11, 12)
        cube1 = new_cube(
            width=1000,
            height=1000,
            variables=dict(chl=0.6, tsm=0.9, flags=16),
            x_start=x1,
            y_start=y1,
            x_res=(x2 - x1) / 1000,
            y_res=(x2 - x1) / 1000,
        )
        cube1.attrs = {}

        md_adjuster = CubeMetadataAdjuster()
        cube2, gm, cc = md_adjuster.transform_cube(
            cube1,
            GridMapping.from_dataset(cube1),
            CubeConfig(metadata=dict(title='S2L2A subset'),
                       variable_metadata=dict(
                           chl=dict(long_name='Chlorophyll'),
                           tsm=dict(long_name='Total suspended matter'),
                           flags=dict(long_name='Quality flags'),
                       ))
        )
        self.assertIsNot(cube1, cube2)
        self.assertIsInstance(gm, GridMapping)
        self.assertIsInstance(cc, CubeConfig)

        date_created = cube2.attrs.pop('date_created', None)
        self.assertIsInstance(date_created, str)
        history = cube2.attrs.pop('history', None)
        self.assertIsInstance(history, list)

        self.assertEqual(
            {
                'Conventions': 'CF-1.7',
                'title': 'S2L2A subset',
                'geospatial_bounds_crs': 'CRS84',
                'geospatial_bounds': 'POLYGON((53 11, 53 12,'
                                     ' 54 12, 54 11, 53 11))',
                'geospatial_lat_max': 12,
                'geospatial_lat_min': 11,
                'geospatial_lat_resolution': 0.001,
                'geospatial_lat_units': 'degrees_north',
                'geospatial_lon_max': 54,
                'geospatial_lon_min': 53,
                'geospatial_lon_resolution': 0.001,
                'geospatial_lon_units': 'degrees_east',
            },
            cube2.attrs)
        self.assertEqual({'long_name': 'Chlorophyll'},
                         cube2.chl.attrs)
        self.assertEqual({'long_name': 'Total suspended matter'},
                         cube2.tsm.attrs)
        self.assertEqual({'long_name': 'Quality flags'},
                         cube2.flags.attrs)

    def test_metadata_adjusted_other_crs(self):
        crs = pyproj.CRS.from_string('epsg:25832')
        t = pyproj.Transformer.from_crs(crs_from=CRS_CRS84, crs_to=crs)
        ((x1, x2), (y1, y2)) = t.transform((53, 54), (10, 11))
        cube1 = new_cube(
            width=1000,
            height=1000,
            variables=dict(chl=0.6, tsm=0.9, flags=16),
            x_start=x1,
            y_start=y1,
            x_res=(x2 - x1) / 1000,
            y_res=(x2 - x1) / 1000,
            x_name='x',
            y_name='y',
            crs=crs,
        )
        cube1.attrs = {}

        md_adjuster = CubeMetadataAdjuster()
        cube2, gm, cc = md_adjuster.transform_cube(
            cube1,
            GridMapping.from_dataset(cube1),
            CubeConfig(metadata=dict(title='S2L2A subset'),
                       variable_metadata=dict(
                           chl=dict(long_name='Chlorophyll'),
                           tsm=dict(long_name='Total suspended matter'),
                           flags=dict(long_name='Quality flags'),
                       ))
        )
        self.assertIsNot(cube1, cube2)
        self.assertIsInstance(gm, GridMapping)
        self.assertIsInstance(cc, CubeConfig)

        self.assertAlmostEqual(53.,
                               cube2.attrs.get('geospatial_lon_min'),
                               delta=0.001)
        self.assertAlmostEqual(10.,
                               cube2.attrs.get('geospatial_lat_min'),
                               delta=0.001)
        self.assertAlmostEqual(0.001,
                               cube2.attrs.get('geospatial_lon_resolution'),
                               delta=0.0005)
        self.assertAlmostEqual(0.001,
                               cube2.attrs.get('geospatial_lat_resolution'),
                               delta=0.0005)

    def test_easter_egg(self):
        cube = new_cube()
        md_adjuster = CubeMetadataAdjuster()
        with self.assertRaises(ValueError) as cm:
            md_adjuster.transform_cube(
                cube,
                GridMapping.from_dataset(cube),
                CubeConfig(metadata=dict(
                    inverse_fine_structure_constant=136
                ))
            )
        self.assertEqual(('inverse_fine_structure_constant must be 137'
                          ' or running in wrong universe',),
                         cm.exception.args)
