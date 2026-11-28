import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from dms.desktop import MainWindow, SettingsDialog
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE and platform.system() == "Windows",
                     "desktop UI smoke runs on the Windows release target")
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_builds_at_supported_minimum_size(self):
        with TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"LOCALAPPDATA": directory}), \
             patch("dms.desktop.CameraService.enumerate", return_value=[0]):
            window = MainWindow()
            self.assertEqual(window.camera_combo.count(), 1)
            self.assertGreaterEqual(window.width(), 1100)
            self.assertFalse(window.phone_card.isVisible())
            window.close()

    def test_experimental_phone_is_explicit_setting(self):
        from dms.services import ProductSettings
        dialog = SettingsDialog(ProductSettings(experimental_phone_enabled=False))
        self.assertFalse(dialog.experimental.isChecked())
        self.assertIn("시험 기능", dialog.experimental.text())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
