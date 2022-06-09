import unittest

import click
import xarray as xr
import yaml

from test.cli.helpers import CliTest
from xcube.cli.patch import parse_metadata
from xcube.core.dsio import rimraf
from xcube.core.new import new_cube

TEST_METADATA_FILE = 'patch.yaml'
TEST_OPTIONS_FILE = 'options.yaml'


class PatchTest(CliTest):

    def test_help_option(self):
        result = self.invoke_cli(['patch', '--help'])
        self.assertEqual(0, result.exit_code)

    def test_missing_cube_arg(self):
        result = self.invoke_cli(['patch'])
        self.assertEqual(2, result.exit_code)


class PatchZarrCubeTest(CliTest):
    CUBE_PATH = 'cube.zarr'
    METADATA_PATH = 'metadata.yaml'
    OPTIONS_PATH = 'options.yaml'

    def test_ok(self):
        self.write_cube()
        self.write_metadata({
            "zarr_consolidated_format": 1,
            "metadata": {
                ".zattrs": {
                    "title": "Humpty-Dumpty",
                    "TileSize": "__delete__",   # SNAP adds this, remove!
                },
                "conc_chl/.zattrs": {
                    "title": "CHL"
                },
                "conc_tsm/.zattrs": {
                    "title": "TSM"
                }
            }
        })
        self.write_options({})
        result = self.invoke_cli([
            'patch', self.CUBE_PATH,
            '--metadata', self.METADATA_PATH,
            '--options', self.OPTIONS_PATH,
        ])
        self.assertEqual(0, result.exit_code)
        cube = xr.open_zarr(self.CUBE_PATH)
        self.assertEqual(
            {'Conventions': 'CF-1.7',
             'geospatial_lat_max': 90.0,
             'geospatial_lat_min': -90.0,
             'geospatial_lat_units': 'degrees_north',
             'geospatial_lon_max': 180.0,
             'geospatial_lon_min': -180.0,
             'geospatial_lon_units': 'degrees_east',
             'time_coverage_end': '2010-01-02T00:00:00',
             'time_coverage_start': '2010-01-01T00:00:00',
             'title': 'Humpty-Dumpty'},
            cube.attrs
        )
        self.assertEqual(
            {'long_name': 'Chlorophyll concentration',
             'title': 'CHL',
             'units': 'mg m^-3'},
            cube.conc_chl.attrs
        )
        self.assertEqual(
            {'long_name': 'Total suspended matter dry weight concentration',
             'title': 'TSM',
             'units': 'g m^-3'},
            cube.conc_tsm.attrs
        )

    def test_missing_metadata_option(self):
        result = self.invoke_cli(['patch', self.CUBE_PATH])
        self.assertEqual(1, result.exit_code)

    @classmethod
    def write_options(cls, options):
        if options is not None:
            with open(cls.OPTIONS_PATH, 'w') as fp:
                yaml.dump(options, fp, indent=2)

    @classmethod
    def write_metadata(cls, metadata):
        if metadata is not None:
            with open(cls.METADATA_PATH, 'w') as fp:
                yaml.dump(metadata, fp, indent=2)

    @classmethod
    def write_cube(cls):
        cube = new_cube(width=36,
                        height=18,
                        x_res=10,
                        time_periods=1,
                        drop_bounds=True,
                        variables=dict(
                            conc_chl=0.8,
                            conc_tsm=4.2,
                        ))
        cube.conc_chl.attrs.update({
            "long_name": "Chlorophyll concentration",
            "units": "mg m^-3"
        })
        cube.conc_tsm.attrs.update({
            "long_name": "Total suspended matter dry weight concentration",
            "units": "g m^-3",
        })
        cube.attrs.update({
            "Conventions": "CF-1.7",
            "coordinates": "time lat lon",
            "TileSize": "1024:1024"  # SNAP adds this to NetCDFs
        })
        cube.to_zarr(cls.CUBE_PATH)

    def tearDown(self) -> None:
        rimraf(self.CUBE_PATH,
               self.METADATA_PATH,
               self.OPTIONS_PATH)


class MetadataTest(unittest.TestCase):
    def test_parse_metadata_ok(self):
        metadata = parse_metadata(
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zattrs": {
                        "title": "Humpty-Dumpty"
                    }
                }
            }
        )
        self.assertEqual(
            {
                ".zattrs": {
                    "title": "Humpty-Dumpty"
                }
            },
            metadata
        )

    def test_parse_metadata_fails(self):
        with self.assertRaises(click.ClickException) as cm:
            parse_metadata([])
        self.assertEqual('Invalid consolidated metadata format',
                         f'{cm.exception}')

        with self.assertRaises(click.ClickException) as cm:
            parse_metadata({
                "zarr_consolidated_format": 3,
                "metadata": {}
            })
        self.assertEqual('Unsupported consolidated metadata version',
                         f'{cm.exception}')

        with self.assertRaises(click.ClickException) as cm:
            parse_metadata({
                "zarr_consolidated_format": 1,
                "metadata": []
            })
        self.assertEqual('Invalid metadata format',
                         f'{cm.exception}')

        with self.assertRaises(click.ClickException) as cm:
            parse_metadata({
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zattrs": []
                }
            })
        self.assertEqual('Invalid metadata format:'
                         ' entry ".zattrs" is not an object',
                         f'{cm.exception}')

        with self.assertRaises(click.ClickException) as cm:
            parse_metadata({
                "zarr_consolidated_format": 1,
                "metadata": {
                }
            })
        self.assertEqual('No metadata provided',
                         f'{cm.exception}')

    def test_parse_metadata_warn(self):
        metadata = parse_metadata(
            {
                "zarr_consolidated_format": 1,
                "metadata": {
                    ".zattrs": {
                        "title": "Humpty-Dumpty"
                    },
                    "i_am_invalid_attrs": {},
                    "i/am/invalid/.zattrs": {}
                }
            }
        )
        self.assertEqual(
            {
                ".zattrs": {
                    "title": "Humpty-Dumpty"
                }
            },
            metadata
        )
