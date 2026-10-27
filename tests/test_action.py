import unittest

from dms.action import ActionSmoother


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


if __name__ == "__main__":
    unittest.main()
