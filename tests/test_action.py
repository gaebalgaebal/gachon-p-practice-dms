import unittest

from dms.action import ActionDetection, ActionSmoother, YOLOAction, load_action_model


class ActionSmootherTests(unittest.TestCase):
    def test_majority_requires_consensus_and_averages_scores(self):
        smoother = ActionSmoother(window=3, consensus=2 / 3, min_samples=3)
        self.assertEqual(smoother.update("phone", {"phone": .8, "normal": .2})[0], "unknown")
        label, scores = smoother.update("normal", {"phone": .4, "normal": .6})
        self.assertEqual(label, "unknown")
        label, scores = smoother.update("phone", {"phone": .9, "normal": .1})
        self.assertEqual(label, "phone")
        self.assertAlmostEqual(scores["phone"], .7)

    def test_reset_removes_stale_action(self):
        smoother = ActionSmoother()
        smoother.update("drink", {"drink": .9})
        smoother.reset()
        self.assertEqual(smoother.current(), ("unknown", {}))

    def test_invalid_settings_and_label_are_rejected(self):
        with self.assertRaises(ValueError):
            ActionSmoother(window=0)
        with self.assertRaises(ValueError):
            ActionSmoother(window=2, min_samples=3)
        with self.assertRaises(ValueError):
            ActionSmoother().update("texting", {})


class YOLOActionTests(unittest.TestCase):
    def test_aggregate_maps_source_classes_to_phone(self):
        label, scores = YOLOAction.aggregate(
            [ActionDetection("phone_call", .82, (1., 2., 3., 4.))], .5)
        self.assertEqual(label, "phone")
        self.assertAlmostEqual(scores["phone"], .82)
        self.assertAlmostEqual(scores["normal"], .18)

    def test_empty_detections_are_normal(self):
        label, scores = YOLOAction.aggregate([], .5)
        self.assertEqual(label, "normal")
        self.assertEqual(scores["phone"], 0.)

    def test_backend_defaults_are_separate(self):
        from unittest.mock import patch
        with patch("dms.action.YOLOAction") as yolo:
            load_action_model(__import__("pathlib").Path("model.pt"), "yolo11", None)
            self.assertEqual(yolo.call_args.args[1], .25)
        with patch("dms.action.MobileNetAction") as mobilenet:
            load_action_model(__import__("pathlib").Path("model.pt"), "mobilenet", None)
            self.assertEqual(mobilenet.call_args.args[1], .7)


if __name__ == "__main__":
    unittest.main()
