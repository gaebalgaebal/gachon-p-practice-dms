import importlib.util
import unittest


NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY, "Install NumPy for eye crop tests")
class EyeAdapterTests(unittest.TestCase):
    def test_eye_scores_average_both_eyes(self):
        from dms.eye import EyeScores
        scores = EyeScores(.8, .2, .6, .4)
        self.assertAlmostEqual(scores.closed_score, .3)

    def test_extract_eye_patch_is_square_and_clipped(self):
        import numpy as np
        from dms.eye import extract_eye_patch

        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        points = np.array([(2, 10), (5, 8), (7, 8), (12, 10), (7, 12), (5, 12)])
        patch = extract_eye_patch(frame, points, range(6))
        self.assertIsNotNone(patch)
        self.assertEqual(patch.ndim, 3)
        self.assertGreaterEqual(min(patch.shape[:2]), 4)

    def test_extract_eye_patch_rejects_invalid_geometry(self):
        import numpy as np
        from dms.eye import extract_eye_patch

        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        self.assertIsNone(extract_eye_patch(frame, np.zeros((2, 2)), range(6)))
        with self.assertRaises(ValueError):
            extract_eye_patch(frame, np.zeros((6, 2)), range(6), padding=1)
