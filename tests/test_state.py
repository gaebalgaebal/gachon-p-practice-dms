import math
import unittest

from dms.contracts import Features
from dms.state import Config, StateEngine


class StateTests(unittest.TestCase):
    def test_short_blink_does_not_trigger(self):
        engine = StateEngine()
        engine.update(Features(0, .15, 0))
        self.assertEqual(engine.update(Features(.5, .3, 0)).drowsiness, "normal")

    def test_sustained_closure_and_recovery(self):
        engine = StateEngine()
        for t in [0, .5, 1]:
            self.assertEqual(engine.update(Features(t, .15, 0)).drowsiness, "normal")
        self.assertEqual(engine.update(Features(1.5, .15, 0)).drowsiness, "suspected")
        self.assertEqual(engine.update(Features(2, .3, 0)).drowsiness, "normal")

    def test_hysteresis(self):
        engine = StateEngine()
        engine.update(Features(0, .15, 0))
        for t in [.5, 1, 1.5]:
            state = engine.update(Features(t, .22, 0))
        self.assertEqual(state.drowsiness, "suspected")

    def test_initial_hysteresis_band_is_unknown(self):
        self.assertEqual(StateEngine().update(Features(0, .22, 0)).drowsiness, "unknown")

    def test_face_loss_clears_history(self):
        engine = StateEngine()
        engine.update(Features(0, .15, 40))
        state = engine.update(Features(.5, None, None, False))
        self.assertEqual((state.drowsiness, state.attention), ("unknown", "unknown"))
        self.assertEqual(engine.update(Features(1, .15, 40)).drowsiness, "normal")

    def test_gap_does_not_count_as_closed_time(self):
        engine = StateEngine()
        engine.update(Features(0, .15, 40))
        state = engine.update(Features(5, .15, 40))
        self.assertEqual((state.drowsiness, state.attention), ("normal", "unknown"))

    def test_independent_outputs(self):
        engine = StateEngine()
        for t in [0, .5, 1, 1.5]:
            state = engine.update(Features(t, .3, 40))
        self.assertEqual((state.drowsiness, state.attention, state.action),
                         ("normal", "off_forward", "unknown"))

    def test_invalid_feature_does_not_become_normal(self):
        state = StateEngine().update(Features(0, math.nan, math.inf))
        self.assertEqual((state.drowsiness, state.attention), ("unknown", "unknown"))

    def test_timestamp_rejection(self):
        engine = StateEngine()
        engine.update(Features(1, .3, 0))
        for t in [1, 0, math.nan, math.inf, -1]:
            with self.assertRaises(ValueError):
                engine.update(Features(t, .3, 0))

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            Config(ear_close=.3, ear_open=.2)
        with self.assertRaises(ValueError):
            Config(closed_seconds=-1)

    def test_pitch_detects_sustained_downward_head(self):
        engine = StateEngine()
        for t in [0, .5, 1, 1.5]:
            result = engine.update(Features(t, .3, 0, True, 30))
        self.assertEqual(result.attention, 'off_forward')

    def test_invalid_pitch_is_unknown(self):
        result = StateEngine().update(Features(0, .3, 0, True, math.nan))
        self.assertEqual(result.attention, 'unknown')

    def test_sustained_eye_deviation_detects_off_forward(self):
        engine = StateEngine()
        for t in [0, .5, 1, 1.5]:
            result = engine.update(Features(t, .3, 0, True, 0, .2, 0))
        self.assertEqual(result.attention, "off_forward")

    def test_short_eye_deviation_is_pending_not_forward(self):
        engine = StateEngine()
        self.assertEqual(engine.update(Features(0, .3, 0, True, 0, .2, 0)).attention,
                         "unknown")
        self.assertEqual(engine.update(Features(.5, .3, 0, True, 0, 0, 0)).attention,
                         "forward")

    def test_invalid_gaze_is_unknown_but_missing_gaze_uses_head_pose(self):
        engine = StateEngine()
        self.assertEqual(engine.update(Features(0, .3, 0, True, 0)).attention, "forward")
        self.assertEqual(engine.update(Features(.5, .3, 0, True, 0, math.nan, 0)).attention,
                         "unknown")
