import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))


class UnbiasedOnlyTests(unittest.TestCase):
    def test_old_auxiliary_module_file_removed(self):
        retired_module = "per" + "ception.py"
        self.assertFalse((LIB / retired_module).exists())

    def test_old_merge_symbols_removed(self):
        import merge

        prefix = "E" + "ta"
        alias_prefix = "Per" + "ception"
        function_prefix = "e" + "ta"
        forbidden = (
            f"{prefix}GeometricMerge",
            f"{prefix}ArithmeticMerge",
            f"{prefix}PPMMerge",
            f"{alias_prefix}GeometricMerge",
            f"{alias_prefix}ArithmeticMerge",
            f"{alias_prefix}PPMMerge",
            f"{function_prefix}_expansion_dists",
        )
        for name in forbidden:
            self.assertFalse(hasattr(merge, name), name)

    def test_model_constructor_has_no_old_parameters(self):
        from model import GraphIDYOMModel

        prefix = "e" + "ta"
        params = inspect.signature(GraphIDYOMModel.__init__).parameters
        for name in (f"{prefix}_ltm", f"{prefix}_stm", f"{prefix}_max_depth"):
            self.assertNotIn(name, params)

    def test_plain_ppm_accepts_model_kwargs(self):
        from merge import PPMMerge

        dist = PPMMerge().dist_from_counts(
            [{"a": 2.0, "b": 1.0}, {"a": 3.0, "c": 1.0}],
            alphabet=["a", "b", "c"],
            excluded_count=1,
            exclusion=True,
            update_exclusion=False,
        )

        self.assertEqual(set(dist), {"a", "b", "c"})
        self.assertAlmostEqual(sum(dist.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
