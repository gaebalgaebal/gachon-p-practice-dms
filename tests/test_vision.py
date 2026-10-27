import importlib.util
import unittest

VISION = all(importlib.util.find_spec(name) is not None for name in ('cv2', 'mediapipe', 'numpy'))


@unittest.skipUnless(VISION, 'Install .[vision] for geometry tests')
class VisionTests(unittest.TestCase):
    def test_ear_open_and_closed(self):
        from dms.vision import eye_aspect_ratio
        self.assertAlmostEqual(eye_aspect_ratio([(0, 0), (1, 1), (3, 1), (4, 0),
                                                (3, -1), (1, -1)]), .5)
        self.assertEqual(eye_aspect_ratio([(0, 0), (1, 0), (3, 0), (4, 0),
                                         (3, 0), (1, 0)]), 0)
        self.assertIsNone(eye_aspect_ratio([(0, 0)] * 6))

    def test_rotation_convention(self):
        import numpy as np
        from dms.vision import head_angles
        theta = np.radians(30)
        matrix = np.eye(4)
        matrix[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]]
        yaw, pitch, roll = head_angles(matrix)
        self.assertAlmostEqual(yaw, 30)
        self.assertAlmostEqual(pitch, 0)
        self.assertAlmostEqual(roll, 0)

    def test_iris_position_uses_screen_coordinates_for_both_eyes(self):
        import numpy as np
        from dms.vision import iris_position
        points = np.zeros((478, 2), dtype=float)
        points[33], points[133] = (10, 10), (30, 10)
        points[159], points[145], points[468] = (20, 6), (20, 14), (15, 12)
        # Right-eye semantic corner order is reversed; screen normalization must not be.
        points[362], points[263] = (70, 10), (50, 10)
        points[386], points[374], points[473] = (60, 6), (60, 14), (55, 12)
        x, y = iris_position(points)
        self.assertAlmostEqual(x, .25)
        self.assertAlmostEqual(y, .75)

    def test_blank_image_has_no_face(self):
        import os
        from pathlib import Path
        import numpy as np
        from dms.vision import MediaPipeFaceAnalyzer
        model = Path(os.environ.get('DMS_TEST_MODEL', 'models/face_landmarker.task'))
        if not model.exists():
            self.skipTest('Download model to run blank-frame inference')
        detector = MediaPipeFaceAnalyzer(model)
        try:
            result = detector.analyze(np.zeros((480, 640, 3), dtype=np.uint8), 0)
            self.assertFalse(result.valid)
            self.assertEqual(result.reason, 'no_face')
        finally:
            detector.close()
