"""Select an operational YOLO confidence threshold on validation only."""

import argparse
import json
from pathlib import Path


def tune(args):
    from ultralytics import YOLO

    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError("Checkpoint and data YAML are required")
    model = YOLO(str(args.checkpoint))
    rows = []
    for threshold in args.thresholds:
        result = model.val(
            data=str(args.data.resolve()), split="val", conf=threshold,
            iou=args.nms_iou, imgsz=args.image_size, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, verbose=False,
            project=str(args.project.resolve()), name=f"conf_{threshold:.2f}", exist_ok=True,
        )
        precision, recall = float(result.box.mp), float(result.box.mr)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        rows.append({
            "threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "map50": float(result.box.map50), "map50_95": float(result.box.map),
        })
        print(f"conf={threshold:.2f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    eligible = [row for row in rows if row["recall"] >= args.min_recall]
    candidates = eligible or rows
    selected = max(candidates, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    payload = {
        "selection_split": "validation", "selection_rule": "max_f1_with_min_recall",
        "min_recall": args.min_recall, "selected": selected, "candidates": rows,
        "checkpoint": str(args.checkpoint), "data": str(args.data),
        "image_size": args.image_size, "batch_size": args.batch_size,
        "nms_iou": args.nms_iou,
        "temporal_vote_note": "Representative still frames cannot validate temporal voting.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selected conf={selected['threshold']:.2f}; saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/aihub_abnormal_yolo11n.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/aihub_abnormal_yolo/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/aihub_abnormal_yolo11n.threshold.json"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.25, .35, .45, .55, .65, .75])
    parser.add_argument("--min-recall", type=float, default=.6)
    parser.add_argument("--nms-iou", type=float, default=.7)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo_threshold"))
    args = parser.parse_args()
    if (not args.thresholds or any(not 0 < value <= 1 for value in args.thresholds)
            or not 0 <= args.min_recall <= 1 or not 0 < args.nms_iou <= 1):
        parser.error("Thresholds/NMS IoU must be in (0,1], min-recall in [0,1]")
    try:
        tune(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Threshold tuning error: {exc}\n")


if __name__ == "__main__":
    main()
