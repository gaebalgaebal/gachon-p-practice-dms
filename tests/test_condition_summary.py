import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SPEC = importlib.util.spec_from_file_location(
    "condition_summary", Path(__file__).parents[1] / "scripts" / "summarize_condition_runs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConditionSummaryTests(unittest.TestCase):
    def test_summarizes_coverage_and_failures(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "summary.json").write_text(json.dumps({
                "condition": "glasses", "processing_fps": 15,
                "steady_state_fps": 16, "camera_frames_skipped": 2,
            }), encoding="utf-8")
            rows = [
                {"face_valid": True, "calibrated": True, "reason": "ok",
                 "drowsiness": "normal", "attention": "forward", "brightness_mean": 80},
                {"face_valid": False, "calibrated": True, "reason": "no_face",
                 "drowsiness": "unknown", "attention": "unknown", "brightness_mean": 60},
            ]
            (path / "frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            report = MODULE.summarize_run(path)
            self.assertEqual(report["condition"], "glasses")
            self.assertEqual(report["face_valid_fraction"], .5)
            self.assertEqual(report["any_unknown_fraction"], .5)
            self.assertEqual(report["brightness_mean"], 70)
            self.assertEqual(report["face_failure_reasons"], {"ok": 1, "no_face": 1})


if __name__ == "__main__":
    unittest.main()
