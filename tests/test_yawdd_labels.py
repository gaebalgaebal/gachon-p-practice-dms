import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


root = Path(__file__).resolve().parents[1]
eval_spec = importlib.util.spec_from_file_location(
    "evaluate_yawdd_mar", root / "scripts/evaluate_yawdd_mar.py"
)
evaluator = importlib.util.module_from_spec(eval_spec)
eval_spec.loader.exec_module(evaluator)

import_spec = importlib.util.spec_from_file_location(
    "import_yawdd_labels", root / "scripts/import_yawdd_labels.py"
)
importer = importlib.util.module_from_spec(import_spec)
import_spec.loader.exec_module(importer)


class YawDDLabelEvaluationTests(unittest.TestCase):
    def test_predict_applies_duration_and_hysteresis(self):
        samples = [{"t": i * .2, "mar": value, "split": "val"}
                   for i, value in enumerate([.3, .6, .6, .6, .5, .35])]
        states, events = evaluator.predict(samples, .55, .40, .4)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0][0], .2)
        self.assertEqual(states[-1], "normal")

    def test_evaluate_config_counts_unknown_as_error(self):
        grouped = {"v": [{"t": 0., "mar": None, "split": "val"},
                         {"t": 1., "mar": .6, "split": "val"},
                         {"t": 2., "mar": .6, "split": "val"}]}
        report = evaluator.evaluate_config(grouped, {"v": [(1., 2.)]},
                                           "val", .55, .40, .5)
        self.assertAlmostEqual(report["frame"]["coverage"], 2 / 3)
        self.assertEqual(report["events"]["matched"], 1)

    def test_load_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "samples.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=evaluator.SAMPLE_FIELDS)
                writer.writeheader()
                writer.writerow({"video_id": "v", "split": "test", "timestamp_s": "0.2",
                                 "mar": "", "valid": "0", "reason": "no_face"})
            rows = evaluator.load_samples(path)
            self.assertIsNone(rows["v"][0]["mar"])

    def test_high_overlap_from_same_source_is_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            root_dir = Path(folder)
            manifest = root_dir / "manifest.csv"
            a_csv = root_dir / "a.csv"
            b_json = root_dir / "b.json"
            output = root_dir / "out.csv"
            manifest_fields = ["sample_id", "file_path", "split", "duration_s"]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest_fields)
                writer.writeheader()
                writer.writerow({"sample_id": "yawdd:video:001", "file_path": "x.avi",
                                 "split": "val", "duration_s": "10"})
            with a_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["interval_id", "video_id",
                    "file_name", "split", "start_s", "end_s", "label", "annotator"])
                writer.writeheader()
            b_json.write_text(json.dumps({"dataset": "YawDD", "annotator": "B",
                "time_unit": "seconds", "summary": {"total_yawn_intervals": 2},
                "videos": [{"video_id": "yawdd:video:001", "file_name": "x.avi",
                    "split": "validation", "review_status": "complete", "yawn_count": 2,
                    "annotations": [
                        {"interval_id": "i1", "label": "yawn", "start_s": 1., "end_s": 3.},
                        {"interval_id": "i2", "label": "yawn", "start_s": 1.1, "end_s": 2.9},
                    ]}]}), encoding="utf-8")
            summary = importer.merge_labels(manifest, a_csv, b_json, output)
            self.assertEqual(summary["merged_overlaps"], 1)
            with output.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["quality_flag"], "merged_overlap")


if __name__ == "__main__":
    unittest.main()
