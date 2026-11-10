import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from dms.yawdd import parse_yawdd_path, split_subjects, validate_interval


label_spec = importlib.util.spec_from_file_location(
    "label_yawdd", Path(__file__).resolve().parents[1] / "scripts/label_yawdd.py"
)
labeler = importlib.util.module_from_spec(label_spec)
label_spec.loader.exec_module(labeler)


class YawDDTests(unittest.TestCase):
    def test_parse_mirror_filename(self):
        item = parse_yawdd_path("YawDD dataset/Mirror/Female_mirror/19-FemaleNoGlasses-Talking&Yawning.avi")
        self.assertEqual(item.subject_id, "yawdd:female:19")
        self.assertEqual((item.camera, item.activity), ("mirror", "talking_yawning"))
        self.assertEqual((item.coarse_label, item.annotation_status),
                         ("yawn", "interval_required"))

    def test_parse_dash_and_double_extension(self):
        item = parse_yawdd_path("YawDD dataset/Dash/Female/11-FemaleGlasses.avi.avi")
        self.assertEqual((item.camera, item.activity, item.coarse_label),
                         ("dash", "sequence", "unknown"))

    def test_gender_is_part_of_subject_identity(self):
        male = parse_yawdd_path("Mirror/Male_mirror Avi Videos/1-MaleNoGlasses-Normal.avi")
        female = parse_yawdd_path("Mirror/Female_mirror/1-FemaleNoGlasses-Normal.avi")
        self.assertNotEqual(male.subject_id, female.subject_id)

    def test_split_is_deterministic(self):
        subjects = [f"yawdd:female:{i:02d}" for i in range(1, 20)]
        self.assertEqual(split_subjects(subjects, seed=3),
                         split_subjects(reversed(subjects), seed=3))

    def test_interval_validation_and_append(self):
        with self.assertRaises(ValueError):
            validate_interval(3, 2, 10)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "intervals.csv"
            first = labeler.append_interval(output, "yawdd:video:001", "a.avi", 1, 2, 10)
            second = labeler.append_interval(output, "yawdd:video:001", "a.avi", 4, 5, 10)
            self.assertNotEqual(first["interval_id"], second["interval_id"])
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
