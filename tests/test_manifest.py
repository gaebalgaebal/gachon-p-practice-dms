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
    def write_manifest(self, path, rows, header=None):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or ["sample_id", "subject_id", "session_id", "split", "label"])
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

    def test_valid_dms_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            header = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
                      "frame_index", "timestamp_s", "task", "label", "split"]
            rows = [
                ["1", "mrl_eye", "a", "sa", "a.png", "0", "0", "eye_state", "open", "train"],
                ["2", "mrl_eye", "b", "sb", "b.png", "0", "0", "eye_state", "closed", "val"],
                ["3", "mrl_eye", "c", "sc", "c.png", "0", "0", "eye_state", "open", "test"],
            ]
            self.write_manifest(path, rows, header)
            self.assertEqual(module.validate(path, "dms"), 3)

    def test_dms_manifest_rejects_dataset_task_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            header = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
                      "frame_index", "timestamp_s", "task", "label", "split"]
            rows = [
                ["1", "mrl_eye", "a", "sa", "a.png", "0", "0", "action", "phone", "train"],
                ["2", "mrl_eye", "b", "sb", "b.png", "0", "0", "eye_state", "closed", "val"],
                ["3", "mrl_eye", "c", "sc", "c.png", "0", "0", "eye_state", "open", "test"],
            ]
            self.write_manifest(path, rows, header)
            with self.assertRaisesRegex(ValueError, "not configured"):
                module.validate(path, "dms")

    def test_dms_manifest_rejects_invalid_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.csv"
            header = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
                      "frame_index", "timestamp_s", "task", "label", "split"]
            rows = [
                ["1", "yawdd", "a", "sa", "a.avi", "-1", "0", "yawn", "normal", "train"],
                ["2", "yawdd", "b", "sb", "b.avi", "0", "0", "yawn", "yawning", "val"],
                ["3", "yawdd", "c", "sc", "c.avi", "0", "0", "yawn", "normal", "test"],
            ]
            self.write_manifest(path, rows, header)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                module.validate(path, "dms")
