"""Download the official MediaPipe model locally; verify the pinned SHA256."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
# Populated from the official versioned asset during integration verification.
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/face_landmarker.task"))
    args = parser.parse_args()
    if args.output.exists():
        data = args.output.read_bytes()
    else:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        parser.error(f"Model checksum mismatch: {digest}; file was not written")
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(f"Verified {args.output}: sha256={digest}")


if __name__ == "__main__":
    main()
