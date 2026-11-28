import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dms.product import DmsEngine
from dms.services import (ProductSettings, SessionService, SettingsService,
                          cleanup_old_sessions, create_diagnostic_report,
                          recover_incomplete_sessions, verify_model_manifest)
from dms.vision import Observation


class FakeAnalyzer:
    def __init__(self, observations):
        self.observations = iter(observations)
        self.closed = False

    def analyze(self, frame, timestamp_s):
        return next(self.observations)

    def close(self):
        self.closed = True


def valid_observation(ear=.3, mar=.2, yaw=0., pitch=0.):
    return Observation(True, "ok", 1, ear, ear, ear, yaw, pitch, 0., .5, .5, None, mar)


class ProductEngineTests(unittest.TestCase):
    def test_calibration_progress_and_operational_warning(self):
        observations = [valid_observation() for _ in range(20)]
        engine = DmsEngine(FakeAnalyzer(observations))
        results = [engine.process(None, index * .15) for index in range(20)]
        self.assertFalse(results[0].calibrated)
        self.assertTrue(results[-1].calibrated)
        self.assertEqual(results[-1].calibration_message, "ready")
        self.assertFalse(results[-1].warning)

    def test_action_model_requires_explicit_experimental_mode(self):
        with self.assertRaisesRegex(ValueError, "experimental_action"):
            DmsEngine(FakeAnalyzer([]), action_model=object())

    def test_recalibration_clears_ready_state(self):
        engine = DmsEngine(FakeAnalyzer([valid_observation() for _ in range(20)]))
        for index in range(20):
            result = engine.process(None, index * .15)
        self.assertTrue(result.calibrated)
        engine.reset_calibration()
        self.assertIsNone(engine.calibrator.neutral)


class ProductServiceTests(unittest.TestCase):
    def test_settings_round_trip_and_unknown_keys_are_ignored(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            service = SettingsService(path)
            settings = ProductSettings(camera_index=2, sound_enabled=False,
                                       output_directory=str(Path(directory) / "out"))
            service.save(settings)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["future_option"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = service.load()
            self.assertEqual(loaded.camera_index, 2)
            self.assertFalse(loaded.sound_enabled)

    def test_session_exports_summary_and_events_without_frames(self):
        class Result:
            calibrated = True
            unknown = False
            timestamp_s = 2.0
            state = type("State", (), {"drowsiness": "suspected", "yawn": "normal",
                                        "attention": "forward"})()

        ticks = iter([10., 15.])
        with TemporaryDirectory() as directory:
            service = SessionService(Path(directory), 0, clock=lambda: next(ticks))
            service.update(Result(), 12.5)
            result = Result(); result.timestamp_s = 3.0
            result.state = type("State", (), {"drowsiness": "normal", "yawn": "normal",
                                               "attention": "forward"})()
            service.update(result, 10.0)
            output = service.finish()
            self.assertTrue((output / "session.json").is_file())
            self.assertTrue((output / "events.csv").is_file())
            self.assertFalse((output / "frames.jsonl").exists())
            payload = json.loads((output / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["events"][0]["kind"], "eyes_closed")

    def test_diagnostic_report_has_hash_but_no_image_payload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.bin"; model.write_bytes(b"model")
            output = create_diagnostic_report(root / "report.json", [model], [0])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"][0]["name"], "model.bin")
            self.assertNotIn("frame", payload)

    def test_incomplete_session_is_recovered(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ticks = iter([10.])
            service = SessionService(root, 0, clock=lambda: next(ticks))
            self.assertTrue((service.session_dir / ".incomplete").exists())
            self.assertEqual(recover_incomplete_sessions(root), 1)
            self.assertTrue((service.session_dir / "recovered_session.json").exists())

    def test_old_sessions_are_removed_by_retention(self):
        import os
        with TemporaryDirectory() as directory:
            root = Path(directory); old = root / "old"; old.mkdir()
            os.utime(old, (0, 0))
            self.assertEqual(cleanup_old_sessions(root, 1, now=200000), 1)

    def test_model_manifest_rejects_modified_model(self):
        import hashlib
        with TemporaryDirectory() as directory:
            root = Path(directory); model = root / "model.bin"; model.write_bytes(b"ok")
            digest = hashlib.sha256(b"ok").hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"models": {"model.bin": digest}}))
            self.assertEqual(verify_model_manifest(manifest, root)["model.bin"], digest)
            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_model_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
