import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from dms.mrl_eye import label_from_scores, parse_mrl_filename, split_subjects


spec = importlib.util.spec_from_file_location(
    "prepare_mrl_eye", Path(__file__).resolve().parents[1] / "scripts/prepare_mrl_eye.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)

threshold_spec = importlib.util.spec_from_file_location(
    "select_eye_threshold",
    Path(__file__).resolve().parents[1] / "scripts/select_eye_threshold.py"
)
thresholds = importlib.util.module_from_spec(threshold_spec)
threshold_spec.loader.exec_module(thresholds)


class MRLEyeTests(unittest.TestCase):
    def test_parse_official_attribute_order(self):
        item = parse_mrl_filename("s0012_00001_0_1_0_2_1_03.png")
        self.assertEqual(item.subject_id, "s0012")
        self.assertEqual(item.label, "closed")
        self.assertEqual((item.glasses, item.reflections, item.sensor_id), (1, 2, "03"))

    def test_invalid_filename_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sensor_id"):
            parse_mrl_filename("s0001_00001_0_0_1_0_1_09.png")

    def test_subject_split_is_deterministic_and_disjoint(self):
        subjects = [f"s{index:04d}" for index in range(1, 38)]
        first = split_subjects(subjects, seed=7)
        second = split_subjects(reversed(subjects), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "val", "test"})

    def test_score_threshold_abstains(self):
        self.assertEqual(label_from_scores(.65, .35, .70), "unknown")
        self.assertEqual(label_from_scores(.8, .2, .70), "open")
        self.assertEqual(label_from_scores(.1, .9, .70), "closed")

    def test_validation_threshold_selection(self):
        rows = [
            {"label": "open", "open_score": ".55", "closed_score": ".45"},
            {"label": "closed", "open_score": ".10", "closed_score": ".90"},
        ]
        result = thresholds.evaluate_thresholds(rows, .50, .60, .10)
        self.assertEqual(result["selected_threshold"], .50)
        self.assertEqual(result["selected_report"]["coverage"], 1.0)

    def test_manifest_conversion_preserves_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "mrl"
            source.mkdir()
            for index in range(1, 7):
                state = index % 2
                (source / f"s{index:04d}_00001_0_0_{state}_0_1_01.png").touch()
            output = root / "manifests" / "mrl.csv"
            self.assertEqual(prepare.build_manifest(source, output), 6)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["dataset"] for row in rows}, {"mrl_eye"})
            self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
            owners = {}
            for row in rows:
                owners.setdefault(row["subject_id"], set()).add(row["split"])
                self.assertIn(row["label"], {"open", "closed"})
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()))


if __name__ == "__main__":
    unittest.main()
