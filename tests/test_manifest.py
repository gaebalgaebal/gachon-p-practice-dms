import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "validate_manifest", Path(__file__).resolve().parents[1] / "scripts/validate_manifest.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ManifestTests(unittest.TestCase):
    def write_manifest(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "subject_id", "session_id", "split", "label"])
            writer.writerows(rows)

    def test_valid_and_leaking_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            rows = [["1", "a", "sa", "train", "normal"],
                    ["2", "b", "sb", "val", "normal"],
                    ["3", "c", "sc", "test", "normal"]]
            self.write_manifest(path, rows)
            self.assertEqual(module.validate(path), 3)
            rows[2][1] = "a"
            self.write_manifest(path, rows)
            with self.assertRaisesRegex(ValueError, "leakage"):
                module.validate(path)

    def test_duplicate_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            self.write_manifest(path, [["1", "a", "sa", "train", "normal"],
                                       ["1", "a", "sa", "train", "normal"]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.validate(path)
