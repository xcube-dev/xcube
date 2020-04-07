from typing import Any
import array

# noinspection PyUnresolvedReferences
class AlmostEqualDeepMixin:

    def assertAlmostEqualDeep(self,
                              first: Any,
                              second: Any,
                              msg: str = None,
                              places: int = None,
                              delta: float = None):
        """
        Performs self.assertAlmostEqual() on number, and numbers in dictionaries and sequences.

        Note that decimal places (from zero) are usually not the same
        as significant digits (measured from the most significant digit).

        If the two objects compare equal then they will automatically
        compare almost equal.
        """
        if first == second:
            # shortcut
            return

        if isinstance(first, float) or isinstance(second, float):
            self.assertAlmostEqual(first, second, msg=msg, places=places, delta=delta)
            # success
            return

        if isinstance(first, set) or isinstance(second, set):
            # we currently cannot almost-equal-compare sets, but it is possible:
            #  - first remove items from both sets that are strictly equal
            #  - secondly remove items from both sets that are almost equal
            #  - are both empty in the end?
            self.assertEqual(first, second, msg=msg)
            # success
            return

        if isinstance(first, dict) and isinstance(second, dict):
            if len(first) != len(second):
                self.assertEqual(first, second, msg=msg)  # fail
            for key, first_value in first.items():
                if key not in second:
                    self.assertEqual(first, second, msg=msg)  # fail
                second_value = second[key]
                self.assertAlmostEqualDeep(first_value, second_value, msg=msg, delta=delta, places=places)
            for key, second_value in second.items():
                if key not in first:
                    self.assertEqual(first, second, msg=msg)  # fail
            # success
            return

        if isinstance(first, (list, tuple, array.array)) and isinstance(second, (list, tuple, array.array)):
            if len(first) != len(second):
                # let fail
                self.assertEqual(first, second, msg=msg)
            for first_value, second_value in zip(first, second):
                self.assertAlmostEqualDeep(first_value, second_value, msg=msg, delta=delta, places=places)
            # success
            return

        self.assertEqual(first, second, msg=msg)
