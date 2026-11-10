import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "evaluate_events", Path(__file__).parents[1] / "scripts" / "evaluate_events.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EventMetricTests(unittest.TestCase):
    def test_match_reports_miss_false_alarm_and_latency(self):
        rows = []
        for index in range(21):
            t = index * .5
            attention = "off_forward" if 2.5 <= t < 4 or 8 <= t < 9 else "forward"
            rows.append({"timestamp_s": t, "attention": attention})
        truth = [MODULE.Event("attention", "off_forward", 2, 4),
                 MODULE.Event("attention", "off_forward", 6, 7)]
        group = MODULE.evaluate(rows, truth, tolerance_s=.1)["groups"]["attention:off_forward"]
        self.assertEqual(group["matched_events"], 1)
        self.assertEqual(group["missed_events"], 1)
        self.assertEqual(group["false_alarm_events"], 1)
        self.assertAlmostEqual(group["detection_latency_s_mean"], .5)
        self.assertAlmostEqual(group["false_alarms_per_minute"], 6)

    def test_prediction_within_tolerance_matches(self):
        target = [MODULE.Event("drowsiness", "suspected", 3, 4)]
        guess = [MODULE.Event("drowsiness", "suspected", 4.2, 5)]
        matches, missed, false_alarms = MODULE.match_events(target, guess, .25)
        self.assertEqual((len(matches), missed, false_alarms), (1, 0, 0))

    def test_embedded_manual_labels_become_task_events(self):
        rows = [
            {"timestamp_s": 0, "ground_truth": "normal"},
            {"timestamp_s": 1, "ground_truth": "eyes_closed"},
            {"timestamp_s": 2, "ground_truth": "eyes_closed"},
            {"timestamp_s": 3, "ground_truth": "normal"},
            {"timestamp_s": 4, "ground_truth": "away"},
            {"timestamp_s": 5, "ground_truth": "normal"},
        ]
        events = MODULE.embedded_truth_events(rows)
        self.assertEqual(events, [
            MODULE.Event("drowsiness", "suspected", 1, 3),
            MODULE.Event("attention", "off_forward", 4, 5),
        ])

    def test_unknown_and_normal_embedded_labels_are_not_events(self):
        rows = [{"timestamp_s": 0, "ground_truth": "unknown"},
                {"timestamp_s": 1, "ground_truth": "normal"}]
        self.assertEqual(MODULE.embedded_truth_events(rows), [])


if __name__ == "__main__":
    unittest.main()
