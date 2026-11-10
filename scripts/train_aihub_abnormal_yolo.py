"""Train and evaluate YOLO11 on the prepared AI Hub abnormal-action subset."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args):
    import torch
    from ultralytics import YOLO

    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}")
    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if str(device) == "0" and not torch.cuda.is_available():
        raise ValueError("CUDA device 0 was requested but CUDA is unavailable")

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.image_size,
        batch=args.batch_size, device=device, workers=args.workers,
        project=str(args.project.resolve()), name=args.name, exist_ok=False,
        seed=args.seed, deterministic=True, patience=args.patience,
        cache=False, plots=True, verbose=True,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not create {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)

    test_model = YOLO(str(args.output))
    test = test_model.val(
        data=str(args.data.resolve()), split="test", imgsz=args.image_size,
        batch=args.batch_size, device=device, workers=args.workers,
        project=str(args.project.resolve()), name=f"{args.name}_test", plots=True,
    )
    class_metrics = {}
    for index, class_id in enumerate(test.box.ap_class_index.tolist()):
        class_metrics[str(test.names[int(class_id)])] = {
            "precision": float(test.box.p[index]), "recall": float(test.box.r[index]),
            "map50": float(test.box.ap50[index]), "map50_95": float(test.box.ap[index]),
        }
    payload = {
        "model": args.model, "checkpoint": str(args.output),
        "checkpoint_sha256": sha256(args.output),
        "data": str(args.data.resolve()), "data_sha256": sha256(args.data),
        "epochs_requested": args.epochs, "image_size": args.image_size,
        "batch_size": args.batch_size, "seed": args.seed, "device": str(device),
        "torch": str(torch.__version__),
        "ultralytics": str(__import__("ultralytics").__version__),
        "train_run": str(Path(result.save_dir)), "test_run": str(Path(test.save_dir)),
        "test": {
            "precision": float(test.box.mp), "recall": float(test.box.mr),
            "map50": float(test.box.map50), "map50_95": float(test.box.map),
            "classes": class_metrics,
        },
        "limitations": [
            "Labels are driver body/action regions, not phone-object boxes.",
            "The source has no driver identity; video_id is used as a split proxy.",
        ],
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output} and {metrics_path}; test mAP50-95={test.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/aihub_abnormal_yolo/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--output", type=Path, default=Path("models/aihub_abnormal_yolo11n.pt"))
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--name", default="aihub_abnormal_yolo11n")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index")
    args = parser.parse_args()
    if min(args.epochs, args.image_size, args.batch_size) <= 0 or args.workers < 0 or args.patience < 0:
        parser.error("epochs/image-size/batch-size must be positive; workers/patience nonnegative")
    try:
        train(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"YOLO training error: {exc}\n")


if __name__ == "__main__":
    main()
