import ast
import csv
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORES = {
    "analysis/code/estimation/nb_core.py": "80671d9b152134d40a6a47b53b2838dc0d63006a2dfd98eecf38fbf6bca090f5",
    "analysis/code/inference/joint_refit.py": "2f147341462a5ce7a063fe0ba5a3ed029b54b9c1deadcd1e180feb38be823c52",
    "analysis/code/inference/weighted_information.py": "4b3b3cbf9f5266d95cd141ce24b1f00adea4078664c99e036d915c2ab639c41b",
}


class PublicLayoutTests(unittest.TestCase):
    def test_numerical_cores_unchanged(self):
        for name, expected in CORES.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_source_inventory(self):
        with (ROOT / "provenance/SOURCE_MANIFEST.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / "analysis").rglob("*")
                    if p.is_file() and p.suffix in (".py", ".R")}
        self.assertEqual({r["path"] for r in rows}, expected)
        for row in rows:
            with self.subTest(name=row["path"]):
                self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["release_sha256"])

    def test_public_source_names(self):
        for path in (ROOT / "analysis").rglob("*"):
            relative = path.relative_to(ROOT).as_posix()
            self.assertTrue(relative.isascii(), relative)
            self.assertNotIn("_yyq", relative)
            self.assertNotIn("data to", relative)
            self.assertNotIn("data_to", relative)

    def test_figure_paths_still_read_frozen_data(self):
        for name in ("make_cross_section_spatial_maps.py", "make_mouse_region_overlays.py",
                     "run_mouse_reference_deconvolution.py", "run_compact_benchmark.py"):
            path = ROOT / "analysis/code/figures" / name
            tree = ast.parse(path.read_text())
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "ROOT" for t in n.targets))
            value = eval(compile(ast.Expression(assignment.value), str(path), "eval"),
                         {"Path": Path, "__file__": str(path)})
            self.assertEqual(value, ROOT / "04_数值实验")


if __name__ == "__main__":
    unittest.main(verbosity=2)
