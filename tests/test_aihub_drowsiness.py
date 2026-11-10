import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from dms.aihub_drowsiness import controlled_labels, eye_label, subject_from_member


spec = importlib.util.spec_from_file_location(
    "prepare_aihub_drowsiness",
    Path(__file__).resolve().parents[1] / "scripts/prepare_aihub_drowsiness.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class AIHubDrowsinessTests(unittest.TestCase):
    def test_label_helpers(self):
        payload = {"ObjectInfo": {"BoundingBox": {
            "Leye": {"isVisible": True, "Opened": False},
            "Reye": {"isVisible": True, "Opened": False},
        }}}
        self.assertEqual(eye_label(payload), "closed")
        self.assertEqual(controlled_labels(
            "001_G1_01_무광원_계기판_졸음재현_20200917_182234_02144.jpg"
        ), ("drowsy", "계기판"))
        self.assertEqual(subject_from_member("001_G1/a.json", "controlled"),
                         "aihub_drowsiness:controlled:001")

    def test_builds_subject_disjoint_archive_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "Training"
            root.mkdir(parents=True)
            source = root / "[원천]bbox(통제환경).zip"
            labels = root / "[라벨]bbox(통제환경).zip"
            with ZipFile(source, "w") as images, ZipFile(labels, "w") as annotations:
                for identity in range(1, 7):
                    name = (f"{identity:03d}_G1_01_무광원_계기판_"
                            f"{'졸음재현' if identity % 2 else '정상주시'}_"
                            "20200917_182234_02144.jpg")
                    member = f"{identity:03d}_G1/{name}"
                    images.writestr(member, b"image")
                    payload = {
                        "FileInfo": {"FileName": name},
                        "UserInfo": {"ID": identity, "Gender": 1, "Age": 30},
                        "Accessory": {"Mask": False, "Glasses": False, "Cap": False},
                        "ObjectInfo": {"BoundingBox": {
                            "Leye": {"isVisible": True, "Opened": identity % 2 == 0},
                            "Reye": {"isVisible": True, "Opened": identity % 2 == 0},
                        }},
                    }
                    annotations.writestr(member.replace(".jpg", ".json"),
                                         json.dumps(payload))
            output = Path(folder) / "manifests" / "aihub.csv"
            self.assertEqual(prepare.build_manifest(Path(folder), output), 18)
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["manifest_rows"], 18)
            self.assertEqual(summary["unmatched_label"], 0)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["task"] for row in rows},
                             {"eye_state", "drowsiness", "head_pose"})
            owners = {}
            for row in rows:
                owners.setdefault(row["subject_id"], set()).add(row["split"])
            self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
            self.assertTrue(all(len(value) == 1 for value in owners.values()))


if __name__ == "__main__":
    unittest.main()
