import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dms.live import label_at_time, load_label_schedule


class LiveLabelScheduleTests(unittest.TestCase):
    def test_schedule_boundaries_and_gaps(self):
        schedule = [(0, 2, "normal"), (3, 5, "phone")]
        self.assertEqual(label_at_time(schedule, 0), "normal")
        self.assertEqual(label_at_time(schedule, 2), "unknown")
        self.assertEqual(label_at_time(schedule, 4), "phone")
        self.assertEqual(label_at_time(schedule, 5), "unknown")

    def test_schedule_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schedule.json"
            path.write_text(json.dumps([
                {"start_s": 0, "end_s": 2, "label": "normal"},
                {"start_s": 2, "end_s": 4, "label": "phone"},
            ]), encoding="utf-8")
            self.assertEqual(load_label_schedule(path),
                             [(0, 2, "normal"), (2, 4, "phone")])

            path.write_text(json.dumps([
                {"start_s": 0, "end_s": 3, "label": "normal"},
                {"start_s": 2, "end_s": 4, "label": "phone"},
            ]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_label_schedule(path)


if __name__ == "__main__":
    unittest.main()
