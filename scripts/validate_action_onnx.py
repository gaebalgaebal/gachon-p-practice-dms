"""Compare PyTorch and OpenVINO action outputs on manifest images."""

import argparse
import csv
import json
import time
from pathlib import Path

from dms.action import MobileNetAction, OpenVINOAction


def validate(torch_checkpoint: Path, onnx_checkpoint: Path, manifest: Path,
             samples: int = 100, device: str = "CPU") -> dict:
    import cv2

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"][:samples]
    if not rows:
        raise ValueError("Manifest has no test samples")
    torch_model = MobileNetAction(torch_checkpoint, threshold=0.0 + 1e-9)
    openvino_model = OpenVINOAction(onnx_checkpoint, threshold=0.0 + 1e-9, device=device)
    max_error, agreements, torch_seconds, openvino_seconds = 0., 0, 0., 0.
    for row in rows:
        path = Path(row["file_path"])
        path = path if path.is_absolute() else manifest.parent / path
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Unreadable image: {path}")
        started = time.perf_counter(); torch_label, torch_scores = torch_model.predict(frame)
        torch_seconds += time.perf_counter() - started
        started = time.perf_counter(); ov_label, ov_scores = openvino_model.predict(frame)
        openvino_seconds += time.perf_counter() - started
        agreements += torch_label == ov_label
        max_error = max(max_error, *(abs(torch_scores[key] - ov_scores[key])
                                     for key in torch_scores))
    count = len(rows)
    return {
        "samples": count, "argmax_agreement": agreements / count,
        "max_probability_abs_error": max_error,
        "pytorch_mean_ms": torch_seconds / count * 1000,
        "openvino_mean_ms": openvino_seconds / count * 1000,
        "openvino_device": device,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_checkpoint", type=Path)
    parser.add_argument("onnx_checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    try:
        result = validate(args.torch_checkpoint, args.onnx_checkpoint, args.manifest,
                          args.samples, args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"ONNX validation error: {exc}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
