import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_release import audit
from scripts.create_release_manifest import create


class ReleaseTests(unittest.TestCase):
    def test_manifest_records_hashes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root / "app.exe").write_bytes(b"binary")
            result = create(root)
            self.assertEqual(result["files"][0]["path"], "app.exe")
            self.assertEqual(len(result["files"][0]["sha256"]), 64)
            self.assertTrue((root / "release-manifest.json").is_file())

    def test_audit_rejects_media_and_private_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); (root / "face.mp4").write_bytes(b"video")
            (root / "settings.json").write_text(json.dumps({"path": "/home/person/data"}))
            errors = audit(root)
            self.assertTrue(any("blocked file type" in item for item in errors))
            self.assertTrue(any("local Linux path" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
