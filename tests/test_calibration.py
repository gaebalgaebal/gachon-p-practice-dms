import unittest

from dms.calibration import Calibrator, EyeHistory, EyeSignalHistory, angle_difference
from dms.state import Config


class CalibrationTests(unittest.TestCase):
    def test_neutral_adjusts_eye_thresholds(self):
        calibration = Calibrator()
        for index in range(22):
            result = calibration.update(index / 10, .3, 12., -3., .48, .52)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.config(Config()).ear_close, .195)
        self.assertAlmostEqual(angle_difference(40, result.yaw), 28)
        self.assertAlmostEqual(result.iris_x, .48)
        self.assertAlmostEqual(result.iris_y, .52)

    def test_unstable_iris_delays_calibration(self):
        calibration = Calibrator()
        for index in range(22):
            result = calibration.update(index / 10, .3, 0., 0., index % 2, index % 2)
        self.assertIsNone(result)

    def test_missing_iris_allows_head_only_fallback(self):
        calibration = Calibrator()
        for index in range(22):
            result = calibration.update(index / 10, .3, 0., 0.)
        self.assertIsNotNone(result)
        self.assertIsNone(result.iris_x)
        self.assertIsNone(result.iris_y)

    def test_moving_head_not_calibrated(self):
        calibration = Calibrator()
        for index in range(25):
            calibration.update(index / 10, .3, 20 * (-1) ** index, 0.)
        self.assertIsNone(calibration.neutral)

    def test_loss_and_gap_interrupt_initial_calibration(self):
        calibration = Calibrator()
        for index in range(15):
            calibration.update(index / 10, .3, 0., 0.)
        calibration.update(1.5, None, None, None)
        self.assertEqual(calibration.samples, [])
        calibration.update(2., .3, 0., 0.)
        calibration.update(5., .3, 0., 0.)
        self.assertEqual(len(calibration.samples), 1)

    def test_weighted_history_and_gap(self):
        history = EyeHistory(window_seconds=2)
        history.update(0, True)
        history.update(.5, False)
        fraction, coverage = history.update(1., False)
        self.assertAlmostEqual(fraction, .5)
        self.assertAlmostEqual(coverage, .5)
        self.assertEqual(history.update(2., True), (None, 0.))
        self.assertEqual(history.update(2.5, None), (None, 0.))

    def test_wrapped_angle(self):
        self.assertEqual(angle_difference(-179, 179), 2)

    def test_eye_signal_history_window_and_reset(self):
        history = EyeSignalHistory(window_seconds=1, max_gap=.75)
        self.assertEqual(history.update(0, .3, .31), [(0, .3, .31)])
        samples = history.update(.5, .2, .21)
        self.assertEqual(len(samples), 2)
        samples = history.update(1.2, .25, .26)
        self.assertEqual([row[0] for row in samples], [.5, 1.2])
        self.assertEqual(history.update(2.2, .3, .3), [(2.2, .3, .3)])
        self.assertEqual(history.update(2.3, None, .3), [])
