"""Export a trained MobileNetV3 action checkpoint to ONNX with runtime metadata."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint: Path, output: Path, opset: int = 17) -> dict:
    import torch
    from torchvision.models import mobilenet_v3_small

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() or output.with_suffix(output.suffix + ".metadata.json").exists():
        raise ValueError(f"Output or metadata already exists: {output}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classes = payload.get("classes")
    if (payload.get("architecture") != "mobilenet_v3_small"
            or payload.get("preprocessing") != "resize256_center224_imagenet"
            or payload.get("trained_steps", 0) < 1 or payload.get("smoke_only", False)
            or not isinstance(classes, list) or len(classes) < 2):
        raise ValueError("Require a trained MobileNetV3-Small action checkpoint")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), output,
                      input_names=["images"], output_names=["logits"], opset_version=opset,
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    metadata = {
        "architecture": "mobilenet_v3_small", "classes": classes,
        "preprocessing": payload["preprocessing"], "opset": opset,
        "source_checkpoint_sha256": sha256(checkpoint), "onnx_sha256": sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    if args.opset < 13:
        parser.error("--opset must be at least 13")
    try:
        result = export(args.checkpoint, args.output, args.opset)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"ONNX export error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
