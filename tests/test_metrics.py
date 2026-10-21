import unittest

from dms.metrics import classification_report
from dms.live import parse_roi


class MetricTests(unittest.TestCase):
    def test_unknown_is_false_negative_and_lowers_coverage(self):
        result = classification_report(['normal', 'suspected'], ['normal', 'unknown'],
                                       ['normal', 'suspected'])
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['per_class']['suspected']['recall'], 0)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [0, 0, 1]])

    def test_invalid_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            classification_report(['normal'], ['typo'], ['normal'])

    def test_roi_bounds(self):
        self.assertEqual(parse_roi('0.1,0.2,0.5,0.6'), (.1, .2, .5, .6))
        import argparse
        for value in ['0,0,2,1', 'nan,0,1,1', '-1,0,1,1', '0,0,0,1']:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_roi(value)
