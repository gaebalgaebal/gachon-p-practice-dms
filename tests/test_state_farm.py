import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

from dms.state_farm import map_source_class, split_subjects


spec = importlib.util.spec_from_file_location(
    "prepare_state_farm", Path(__file__).resolve().parents[1] / "scripts/prepare_state_farm.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class StateFarmTests(unittest.TestCase):
    def test_project_action_mapping(self):
        self.assertEqual(map_source_class("c0").action_label, "normal")
        self.assertEqual(map_source_class("c1").action_label, "phone")
        self.assertEqual(map_source_class("c4").action_label, "phone")
        self.assertEqual(map_source_class("c6").action_label, "drink")
        self.assertIsNone(map_source_class("c9").action_label)

    def test_split_is_deterministic_and_disjoint(self):
        subjects = [f"p{i:03d}" for i in range(26)]
        self.assertEqual(split_subjects(subjects, seed=8),
                         split_subjects(reversed(subjects), seed=8))
        self.assertEqual(set(split_subjects(subjects).values()), {"train", "val", "test"})

    def test_manifest_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "state_farm"
            image_root = source / "imgs" / "train"
            rows = []
            for subject_index in range(6):
                for source_class in ("c0", "c1", "c6", "c9"):
                    name = f"img_{subject_index}_{source_class}.jpg"
                    path = image_root / source_class / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    rows.append({"subject": f"p{subject_index:03d}",
                                 "classname": source_class, "img": name})
            with (source / "driver_imgs_list.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject", "classname", "img"])
                writer.writeheader(); writer.writerows(rows)
            output = root / "manifests" / "state_farm.csv"
            summary = prepare.build_manifest(source, output)
            self.assertEqual(summary["samples"], 18)
            self.assertEqual(summary["labels"], {"normal": 6, "phone": 6, "drink": 6})
            with output.open(encoding="utf-8", newline="") as handle:
                converted = list(csv.DictReader(handle))
            owners = {}
            for row in converted:
                owners.setdefault(row["subject_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in owners.values()))


if __name__ == "__main__":
    unittest.main()
