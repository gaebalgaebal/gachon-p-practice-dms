import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from dms.uta_rldd import binary_label, parse_uta_path, split_subjects


eval_spec = importlib.util.spec_from_file_location(
    "evaluate_uta_rldd", Path(__file__).resolve().parents[1] / "scripts/evaluate_uta_rldd.py"
)
evaluator = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluator)


class UTARLDDTests(unittest.TestCase):
    def test_parse_levels_and_split_segment(self):
        normal = parse_uta_path("Fold1_part1/01/0.MP4")
        split = parse_uta_path("Fold3_part2/32/10_2.mp4")
        self.assertEqual((normal.fold, normal.level, normal.label), (1, 0, "alert"))
        self.assertEqual((split.subject_id, split.label, split.segment),
                         ("uta_rldd:32", "drowsy", 2))

    def test_invalid_level_rejected(self):
        with self.assertRaisesRegex(ValueError, "level filename"):
            parse_uta_path("Fold1_part1/01/7.mp4")

    def test_subject_split_is_deterministic(self):
        subjects = [f"uta_rldd:{index:02d}" for index in range(1, 49)]
        self.assertEqual(split_subjects(subjects, seed=5),
                         split_subjects(reversed(subjects), seed=5))

    def test_binary_mapping(self):
        self.assertEqual(binary_label("alert"), "alert")
        self.assertEqual(binary_label("low_vigilance"), "fatigued")
        self.assertEqual(binary_label("unknown"), "unknown")

    def test_evaluator_reports_three_and_two_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "manifest.csv"
            predictions = root / "predictions.csv"
            fields = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
                      "frame_index", "timestamp_s", "task", "label", "split"]
            labels = ["alert", "low_vigilance", "drowsy"]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, label in enumerate(labels):
                    writer.writerow({"sample_id": f"s{index}", "dataset": "uta_rldd",
                                     "subject_id": f"p{index}", "session_id": f"v{index}",
                                     "file_path": "x", "frame_index": 0, "timestamp_s": 0,
                                     "task": "drowsiness", "label": label, "split": "test"})
            with predictions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "prediction"])
                writer.writeheader()
                writer.writerows([{"sample_id": "s0", "prediction": "alert"},
                                  {"sample_id": "s1", "prediction": "drowsy"},
                                  {"sample_id": "s2", "prediction": "unknown"}])
            report = evaluator.evaluate(manifest, predictions)
            self.assertAlmostEqual(report["three_class"]["accuracy"], 1 / 3)
            self.assertAlmostEqual(report["binary_alert_vs_fatigued"]["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
