# Copyright (c) 2018-2024 by xcube team and contributors
# Permissions are hereby granted under the terms of the MIT License:
# https://opensource.org/licenses/MIT.

import unittest

import matplotlib
import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

from test.sampledata import (
    create_highroc_dataset,
    create_c2rcc_flag_var,
    create_cmems_sst_flag_var,
    create_cci_lccs_class_var,
)
from xcube.core.maskset import MaskSet

# noinspection PyProtectedMember
from xcube.core.maskset import _sanitize_flag_values


nan = float("nan")


class MaskSetTest(unittest.TestCase):
    def test_mask_set_with_flag_mask_str(self):
        flag_var = create_cmems_sst_flag_var().chunk(dict(lon=2, lat=2))
        mask_set = MaskSet(flag_var)

        self.assertEqual(4, len(mask_set))
        self.assertEqual(["sea", "land", "lake", "ice"], list(mask_set.keys()))
        self.assertEqual(
            "mask(sea=(1, None), land=(2, None), lake=(4, None), ice=(8, None))",
            str(mask_set),
        )

        expected_chunks = ((1,), (2, 1), (2, 2))
        self.assertMaskOk(mask_set, "sea", expected_chunks)
        self.assertMaskOk(mask_set, "land", expected_chunks)
        self.assertMaskOk(mask_set, "lake", expected_chunks)
        self.assertMaskOk(mask_set, "ice", expected_chunks)

        validation_data = (
            (
                0,
                "sea",
                mask_set.sea,
                np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]], dtype=np.uint8),
            ),
            (
                1,
                "land",
                mask_set.land,
                np.array([[[0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]], dtype=np.uint8),
            ),
            (
                2,
                "lake",
                mask_set.lake,
                np.array([[[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8),
            ),
            (
                3,
                "ice",
                mask_set.ice,
                np.array([[[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]]], dtype=np.uint8),
            ),
        )

        for index, name, mask, data in validation_data:
            self.assertIs(mask, mask_set[index])
            self.assertIs(mask, mask_set[name])
            assert_array_almost_equal(
                mask.values, data, err_msg=f"{index}, {name}, {mask.name}"
            )

    def test_mask_set_with_flag_mask_int_array(self):
        flag_var = create_c2rcc_flag_var().chunk(dict(x=2, y=2))
        mask_set = MaskSet(flag_var)

        self.assertEqual(4, len(mask_set))
        self.assertEqual(["F1", "F2", "F3", "F4"], list(mask_set.keys()))
        self.assertEqual(
            "c2rcc_flags(F1=(1, None), F2=(2, None), F3=(4, None), F4=(8, None))",
            str(mask_set),
        )

        expected_chunks = ((2, 1), (2, 2))
        self.assertMaskOk(mask_set, "F1", expected_chunks)
        self.assertMaskOk(mask_set, "F2", expected_chunks)
        self.assertMaskOk(mask_set, "F3", expected_chunks)
        self.assertMaskOk(mask_set, "F4", expected_chunks)

        validation_data = (
            (
                0,
                "F1",
                mask_set.F1,
                np.array([[1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.uint8),
            ),
            (
                1,
                "F2",
                mask_set.F2,
                np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8),
            ),
            (
                2,
                "F3",
                mask_set.F3,
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            ),
            (
                3,
                "F4",
                mask_set.F4,
                np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8),
            ),
        )

        for index, name, mask, data in validation_data:
            self.assertIs(mask, mask_set[index])
            self.assertIs(mask, mask_set[name])
            assert_array_almost_equal(mask.values, data)

    def assertMaskOk(
        self,
        mask_set: MaskSet,
        mask_name: str,
        expected_chunks: tuple[tuple[int, ...], ...] = None,
    ):
        mask = getattr(mask_set, mask_name)

        self.assertIsInstance(mask, xr.DataArray)

        if expected_chunks:
            import dask.array as da

            self.assertIsInstance(mask.data, da.Array)
        self.assertEqual(expected_chunks, mask.chunks)

        # assert same instance is returned
        self.assertIs(mask, getattr(mask_set, mask_name))

        self.assertIn(mask_name, mask_set)
        # assert same instance is returned
        self.assertIs(mask, mask_set[mask_name])

    def test_get_mask_sets(self):
        dataset = create_highroc_dataset()
        mask_sets = MaskSet.get_mask_sets(dataset)
        self.assertIsNotNone(mask_sets)
        self.assertEqual(len(mask_sets), 1)
        self.assertIn("c2rcc_flags", mask_sets)
        mask_set = mask_sets["c2rcc_flags"]
        self.assertIsInstance(mask_set, MaskSet)

    def test_mask_set_with_flag_values(self):
        s2l2a_slc_meanings = [
            "no_data",
            "saturated_or_defective",
            "dark_area_pixels",
            "cloud_shadows",
            "vegetation",
            "bare_soils",
            "water",
            "clouds_low_probability_or_unclassified",
            "clouds_medium_probability",
            "clouds_high_probability",
            "cirrus",
            "snow_or_ice",
        ]

        data = np.array([[1, 2, 8, 3], [7, 6, 0, 4], [9, 5, 11, 10]], dtype=np.uint8)
        flag_var = xr.DataArray(
            data,
            dims=("y", "x"),
            name="SLC",
            attrs=dict(
                long_name="Scene classification flags",
                flag_values=",".join(f"{i}" for i in range(len(s2l2a_slc_meanings))),
                flag_meanings=" ".join(s2l2a_slc_meanings),
            ),
        )

        mask_set = MaskSet(flag_var)

        self.assertEqual(
            "SLC(no_data=(None, 0), saturated_or_defective=(None, 1), dark_area_pixels=(None, 2), "
            "cloud_shadows=(None, 3), vegetation=(None, 4), bare_soils=(None, 5), water=(None, 6), "
            "clouds_low_probability_or_unclassified=(None, 7), clouds_medium_probability=(None, 8), "
            "clouds_high_probability=(None, 9), cirrus=(None, 10), snow_or_ice=(None, 11))",
            str(mask_set),
        )

        validation_data = (
            (
                0,
                "no_data",
                mask_set.no_data,
                np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8),
            ),
            (
                4,
                "vegetation",
                mask_set.vegetation,
                np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8),
            ),
            (
                10,
                "cirrus",
                mask_set.cirrus,
                np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8),
            ),
            (
                6,
                "water",
                mask_set.water,
                np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            ),
        )

        for index, name, mask, data in validation_data:
            msg = f"index={index}, name={name!r}, data={data!r}"
            self.assertIs(mask, mask_set[index], msg=msg)
            self.assertIs(mask, mask_set[name], msg=msg)
            assert_array_almost_equal(mask.values, data, err_msg=msg)

        self.assertEqual(set(s2l2a_slc_meanings), set(dir(mask_set)))

        html = mask_set._repr_html_()
        self.assertTrue(html.startswith("<html>"))
        self.assertTrue(html.endswith("</html>"))

    def test_mask_set_with_flag_values_as_list(self):
        flag_var = create_cci_lccs_class_var(flag_values_as_list=True)
        mask_set = MaskSet(flag_var)
        self.assertEqual(38, len(mask_set))

    def test_mask_set_with_missing_values_and_masks_attrs(self):
        flag_var = create_c2rcc_flag_var().chunk(dict(x=2, y=2))
        flag_var.attrs.pop("flag_masks", None)
        flag_var.attrs.pop("flag_values", None)
        with self.assertRaises(ValueError):
            MaskSet(flag_var)

    def test_mask_set_with_missing_meanings_attr(self):
        flag_var = create_c2rcc_flag_var().chunk(dict(x=2, y=2))
        flag_var.attrs.pop("flag_meanings", None)
        with self.assertRaises(ValueError):
            MaskSet(flag_var)

    def test_mask_set_get_cmap_without_flag_values(self):
        flag_var = create_c2rcc_flag_var()
        mask_set = MaskSet(flag_var)
        self.assertEqual(4, len(mask_set))
        cmap, norm = mask_set.get_cmap()
        # Uses default "viridis"
        self.assertIsNone(norm)
        self.assertIsInstance(cmap, matplotlib.colors.Colormap)
        self.assertEqual("viridis", cmap.name)

    def test_mask_set_get_cmap_with_flag_values(self):
        flag_var = xr.DataArray(
            [[1, 2], [2, 3]],
            dims=("y", "x"),
            attrs=dict(flag_values="1, 2, 3", flag_meanings="A B C"),
        )
        mask_set = MaskSet(flag_var)
        self.assertEqual(3, len(mask_set))
        # noinspection PyTypeChecker
        cmap, norm = mask_set.get_cmap()
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(norm, matplotlib.colors.BoundaryNorm)
        self.assertEqual("from_maskset", cmap.name)
        colors = cmap(norm(np.array([0, 1, 2, 3])))
        self.assertEqual(4, len(colors))
        np.testing.assert_equal(np.array([0, 0, 0, 0]), colors[0])
        np.testing.assert_equal(np.array([0, 1, 1, 1]), colors[:, 3])

    def test_mask_set_get_cmap_with_flag_values_and_flag_colors(self):
        flag_var = xr.DataArray(
            [[1, 2], [2, 3]],
            dims=("y", "x"),
            attrs=dict(
                flag_values="1, 2, 3",
                flag_colors="red yellow white",
                flag_meanings="A B C",
            ),
            name="quality_flags",
        )
        mask_set = MaskSet(flag_var)
        self.assertEqual(3, len(mask_set))
        # noinspection PyTypeChecker
        cmap, norm = mask_set.get_cmap()
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(norm, matplotlib.colors.BoundaryNorm)
        self.assertEqual("quality_flags", cmap.name)
        colors = cmap(norm(np.array([0, 1, 2, 3])))
        np.testing.assert_equal(
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            ),
            colors,
        )

    def test_mask_set_get_cmap_with_mismatching_flag_values_and_flag_colors(self):
        flag_var = create_cci_lccs_class_var(flag_values_as_list=True)
        mask_set = MaskSet(flag_var)
        self.assertEqual(38, len(mask_set))
        # noinspection PyTypeChecker
        cmap, norm = mask_set.get_cmap()
        self.assertIsInstance(cmap, matplotlib.colors.ListedColormap)
        self.assertIsInstance(norm, matplotlib.colors.BoundaryNorm)
        self.assertEqual("lccs_class", cmap.name)
        self.assertEqual(38, cmap.N)
        colors = cmap(norm(np.array([0, 10, 130, 110, 202, 61, 5])))
        np.testing.assert_almost_equal(
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.39215686, 1.0],
                    [1.0, 0.70588235, 0.19607843, 1.0],
                    [0.74509804, 0.58823529, 0.0, 1.0],
                    [1.0, 0.96078431, 0.84313725, 1.0],
                    [0.0, 0.62745098, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            colors,
        )


class SanitizeFlagValuesTest(unittest.TestCase):
    @staticmethod
    def new_flag_var():
        return xr.DataArray(np.array([[1, 2, 2], [3, 1, 3]]), dims=("y", "x"))

    def test_all_valid(self):
        flag_values = np.array([1, 2, 3])
        var = self.new_flag_var()
        _sanitize_flag_values(var, flag_values)
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertIs(flag_values, flag_values_test)
        self.assertEqual([0, 1, 2], list(index_tracker))

    def test_fill_value_in_encoding(self):
        flag_values = np.array([1, 2, 3])

        var = self.new_flag_var()
        var.encoding["fill_value"] = 1
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([2, 3], list(flag_values_test))
        self.assertEqual([1, 2], list(index_tracker))

        var = self.new_flag_var()
        var.encoding["_FillValue"] = 2
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([1, 3], list(flag_values_test))
        self.assertEqual([0, 2], list(index_tracker))

    def test_fill_value_in_attrs(self):
        flag_values = np.array([1, 2, 3])

        var = self.new_flag_var()
        var.attrs["fill_value"] = 1
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([2, 3], list(flag_values_test))
        self.assertEqual([1, 2], list(index_tracker))

        var = self.new_flag_var()
        var.attrs["_FillValue"] = 2
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([1, 3], list(flag_values_test))
        self.assertEqual([0, 2], list(index_tracker))

    def test_fill_value_is_nan(self):
        flag_values = np.array([1, nan, 3])

        var = self.new_flag_var()
        var.encoding["fill_value"] = nan
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([1, 3], list(flag_values_test))
        self.assertEqual([0, 2], list(index_tracker))

    def test_valid_min_max(self):
        flag_values = np.array([1, 2, 3])

        var = self.new_flag_var()
        var.attrs["valid_min"] = 2
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([2, 3], list(flag_values_test))
        self.assertEqual([1, 2], list(index_tracker))

        var = self.new_flag_var()
        var.attrs["valid_max"] = 2
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([1, 2], list(flag_values_test))
        self.assertEqual([0, 1], list(index_tracker))

        var = self.new_flag_var()
        var.attrs["valid_min"] = 2
        var.attrs["valid_max"] = 2
        flag_values_test, index_tracker = _sanitize_flag_values(var, flag_values)
        self.assertEqual([2], list(flag_values_test))
        self.assertEqual([1], list(index_tracker))
