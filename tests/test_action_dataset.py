import unittest

from dms.action_dataset import sample_times, validate_sessions


class ActionDatasetTests(unittest.TestCase):
    def test_sampling_respects_intervals_and_skips_unknown(self):
        result = list(sample_times([(0, 1, "normal"), (1, 2, "unknown"),
                                    (2, 3, "phone")], 2))
        self.assertEqual(result, [(0., "normal"), (.5, "normal"),
                                  (2., "phone"), (2.5, "phone")])

    def test_subjects_cannot_cross_splits(self):
        sessions = [
            {"subject_id": "a", "session_id": "a1", "split": "train",
             "video": "a.mp4", "schedule": "s.json"},
            {"subject_id": "a", "session_id": "a2", "split": "val",
             "video": "b.mp4", "schedule": "s.json"},
            {"subject_id": "c", "session_id": "c1", "split": "test",
             "video": "c.mp4", "schedule": "s.json"},
        ]
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_sessions(sessions)

    def test_all_splits_are_required(self):
        sessions = [{"subject_id": "a", "session_id": "a1", "split": "train",
                     "video": "a.mp4", "schedule": "s.json"}]
        with self.assertRaisesRegex(ValueError, "train, val and test"):
            validate_sessions(sessions)


if __name__ == "__main__":
    unittest.main()
