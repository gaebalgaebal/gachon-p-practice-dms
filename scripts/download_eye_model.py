"""Download and verify the official Open Model Zoo eye-state ONNX model."""

import argparse
import hashlib
from pathlib import Path
import urllib.request


MODEL_URL = ("https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/"
             "2022.1/open-closed-eye-0001/open_closed_eye.onnx")
MODEL_SHA256 = "4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/open_closed_eye.onnx"))
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
