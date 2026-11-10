import unittest

from dms.aihub_abnormal import category_mapping, split_validation_videos, xywh_to_yolo


class AIHubAbnormalTests(unittest.TestCase):
    def test_category_mapping(self):
        self.assertEqual(category_mapping("차량제어"), ("normal", None, None))
        self.assertEqual(category_mapping("휴대폰조작"), ("phone", "phone_use", 0))
        self.assertEqual(category_mapping("통화"), ("phone", "phone_call", 1))

    def test_validation_video_split_is_disjoint_and_deterministic(self):
        videos = [f"SGA{i:07d}" for i in range(10)]
        first = split_validation_videos(videos, seed=7)
        second = split_validation_videos(reversed(videos), seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"val", "test"})

    def test_xywh_is_clipped_and_normalized(self):
        x, y, w, h = xywh_to_yolo([900, -10, 500, 400], 1280, 720)
        self.assertAlmostEqual(x, (900 + 1280) / (2 * 1280))
        self.assertAlmostEqual(y, 390 / (2 * 720))
        self.assertAlmostEqual(w, 380 / 1280)
        self.assertAlmostEqual(h, 390 / 720)

    def test_invalid_box_rejected(self):
        with self.assertRaises(ValueError):
            xywh_to_yolo([1400, 0, 10, 10], 1280, 720)


if __name__ == "__main__":
    unittest.main()
