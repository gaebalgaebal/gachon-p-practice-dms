"""Evaluate YOLO phone temporal voting on 5-frame AI Hub validation scenes."""

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image

from dms.action import ActionSmoother
from dms.aihub_abnormal import split_validation_videos


PHONE_CATEGORIES = {"휴대폰조작", "통화"}
ALL_CATEGORIES = PHONE_CATEGORIES | {"차량제어"}


def find_one(root: Path, name: str) -> Path:
    found = list(root.rglob(name))
    if len(found) != 1:
        raise ValueError(f"Expected one {name}, found {len(found)}")
    return found[0]


def select_scenes(label_tar: Path, per_category: int, seed: int):
    scenes = []
    with tarfile.open(label_tar) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            payload = json.load(archive.extractfile(member))
            category = (payload.get("scene_info") or {}).get("category_name")
            if category not in ALL_CATEGORIES:
                continue
            video_id = str((payload.get("metadata") or {}).get("video_id", ""))
            scene_id = str((payload.get("scene_info") or {}).get("scene_id", ""))
            frames = (payload.get("scene") or {}).get("data") or []
            if video_id and scene_id and len(frames) >= 4:
                scenes.append({"video_id": video_id, "scene_id": scene_id, "category": category,
                               "members": [f"{video_id}/{scene_id}/img/{row['img_name']}"
                                           for row in frames]})
    assignments = split_validation_videos((row["video_id"] for row in scenes), seed)
    grouped = defaultdict(list)
    for scene in scenes:
        if assignments[scene["video_id"]] == "val":
            grouped[scene["category"]].append(scene)
    rng = random.Random(seed)
    selected = []
    for category in sorted(ALL_CATEGORIES):
        choices = sorted(grouped[category], key=lambda row: row["scene_id"])
        rng.shuffle(choices)
        selected.extend(choices[:per_category])
    return selected


def load_images(source_tar: Path, scenes):
    wanted = {member: (scene["scene_id"], index)
              for scene in scenes for index, member in enumerate(scene["members"])}
    images = {}
    with tarfile.open(source_tar) as archive:
        for member in archive:
            key = wanted.get(member.name)
            if key is not None:
                raw = archive.extractfile(member).read()
                with Image.open(io.BytesIO(raw)) as image:
                    images[key] = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
    return images


def score_config(sequences, threshold, classifier_threshold, window, min_samples, consensus):
    tp = fp = tn = fn = 0
    delays = []
    for sequence in sequences:
        smoother = ActionSmoother(window, consensus, min_samples)
        first_phone = None
        for index, phone_score in enumerate(sequence["scores"]):
            classifier_ok = (classifier_threshold is None
                             or sequence["classifier_scores"][index] >= classifier_threshold)
            raw = "phone" if phone_score >= threshold and classifier_ok else "normal"
            label, _ = smoother.update(raw, {"phone": phone_score, "normal": 1 - phone_score})
            if label == "phone" and first_phone is None:
                first_phone = index
        actual = sequence["phone"]
        predicted = first_phone is not None
        if actual and predicted:
            tp += 1; delays.append(first_phone + 1)
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"threshold": threshold, "classifier_threshold": classifier_threshold,
            "window": window, "min_samples": min_samples,
            "consensus": consensus,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "normal_clip_false_alarm_rate": fp / (fp + tn) if fp + tn else 0.,
            "mean_confirmation_frame": sum(delays) / len(delays) if delays else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(args):
    import cv2
    import torch
    from ultralytics import YOLO

    label_tar, source_tar = find_one(args.source, "VL1.tar"), find_one(args.source, "VS1.tar")
    scenes = select_scenes(label_tar, args.per_category, args.seed)
    images = load_images(source_tar, scenes)
    flat = [(scene, index, images[(scene["scene_id"], index)])
            for scene in scenes for index in range(len(scene["members"]))
            if (scene["scene_id"], index) in images]
    model = YOLO(str(args.checkpoint))
    scores = defaultdict(dict)
    for start in range(0, len(flat), args.batch_size):
        chunk = flat[start:start + args.batch_size]
        results = model.predict([row[2] for row in chunk], conf=min(args.thresholds),
                                device=args.device, verbose=False)
        for (scene, index, _), result in zip(chunk, results):
            value = max(result.boxes.conf.tolist(), default=0.) if result.boxes is not None else 0.
            scores[scene["scene_id"]][index] = value
    classifier_scores = defaultdict(dict)
    classifier_thresholds = [None]
    if args.classifier_checkpoint:
        from torchvision import transforms
        from torchvision.models import mobilenet_v3_small
        payload = torch.load(args.classifier_checkpoint, map_location="cpu", weights_only=True)
        classes = payload["classes"]
        if "phone" not in classes or payload.get("architecture") != "mobilenet_v3_small":
            raise ValueError("Classifier checkpoint must contain a MobileNetV3 phone class")
        classifier_device = f"cuda:{args.device}" if str(args.device).isdigit() else args.device
        classifier = mobilenet_v3_small(weights=None)
        classifier.classifier[-1] = torch.nn.Linear(
            classifier.classifier[-1].in_features, len(classes))
        classifier.load_state_dict(payload["state_dict"])
        classifier.to(classifier_device).eval()
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        for start in range(0, len(flat), args.batch_size):
            chunk = flat[start:start + args.batch_size]
            batch = torch.stack([transform(Image.fromarray(cv2.cvtColor(row[2], cv2.COLOR_BGR2RGB)))
                                 for row in chunk]).to(classifier_device)
            with torch.inference_mode():
                probabilities = classifier(batch).softmax(dim=1)[:, classes.index("phone")].tolist()
            for (scene, index, _), probability in zip(chunk, probabilities):
                classifier_scores[scene["scene_id"]][index] = probability
        classifier_thresholds = args.classifier_thresholds
    sequences, excluded = [], 0
    for scene in scenes:
        ordered = scores[scene["scene_id"]]
        classifier_ordered = classifier_scores[scene["scene_id"]]
        if (len(ordered) != len(scene["members"])
                or args.classifier_checkpoint and len(classifier_ordered) != len(scene["members"])):
            excluded += 1
            continue
        sequences.append({"scene_id": scene["scene_id"], "category": scene["category"],
                          "phone": scene["category"] in PHONE_CATEGORIES,
                          "scores": [ordered[i] for i in range(len(ordered))],
                          "classifier_scores": ([classifier_ordered[i] for i in range(len(ordered))]
                                                if args.classifier_checkpoint else [])})
    candidates = [score_config(sequences, threshold, classifier_threshold,
                               window, minimum, consensus)
                  for threshold in args.thresholds
                  for classifier_threshold in classifier_thresholds
                  for window, minimum, consensus in ((3, 2, .67), (3, 3, .67),
                                                       (5, 2, .6), (5, 3, .6),
                                                       (5, 3, .8), (5, 4, .8))]
    eligible = [row for row in candidates
                if row["recall"] >= args.min_recall
                and row["normal_clip_false_alarm_rate"] <= args.max_false_alarm_rate]
    best_f1 = max(candidates, key=lambda row: (row["f1"], -row["normal_clip_false_alarm_rate"]))
    safe = [row for row in candidates
            if row["normal_clip_false_alarm_rate"] <= args.max_false_alarm_rate]
    best_safe = max(safe, key=lambda row: (row["f1"], row["recall"])) if safe else None
    selected = max(eligible, key=lambda row: (row["f1"], -row["mean_confirmation_frame"])) \
        if eligible else None
    payload = {"split": "validation", "thresholds": args.thresholds, "seed": args.seed,
               "per_category_requested": args.per_category, "scenes": len(sequences),
               "frames": sum(len(row["scores"]) for row in sequences),
               "classifier_checkpoint": (str(args.classifier_checkpoint)
                                         if args.classifier_checkpoint else None),
               "classifier_thresholds": classifier_thresholds,
               "fusion": "YOLO AND MobileNet phone" if args.classifier_checkpoint else "YOLO only",
               "excluded_incomplete_scenes": excluded, "selected": selected,
               "selection_rule": "max_f1_with_min_recall_and_max_normal_clip_false_alarm",
               "min_recall": args.min_recall,
               "max_normal_clip_false_alarm_rate": args.max_false_alarm_rate,
               "eligible_candidates": len(eligible),
               "criteria_met": bool(eligible), "best_f1_unconstrained": best_f1,
               "best_with_false_alarm_constraint_only": best_safe,
               "candidates": candidates,
               "limitations": ["Each scene has only 4-5 labeled frames.",
                               "Scenes start inside the labeled action; delay is confirmation-frame count.",
                               "video_id is a driver-identity proxy."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/aihub_abnormal_yolo11n.pt"))
    parser.add_argument("--classifier-checkpoint", type=Path,
                        help="Optional State Farm MobileNet checkpoint used as an AND gate")
    parser.add_argument("--classifier-thresholds", type=float, nargs="+",
                        default=[.3, .5, .7, .8])
    parser.add_argument("--output", type=Path, default=Path("models/aihub_abnormal_yolo11n.temporal.json"))
    parser.add_argument("--per-category", type=int, default=200)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[.25, .35, .45, .55, .65, .75, .85])
    parser.add_argument("--min-recall", type=float, default=.6)
    parser.add_argument("--max-false-alarm-rate", type=float, default=.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.per_category <= 0 or args.batch_size <= 0
            or any(not 0 < value <= 1 for value in args.thresholds)
            or any(not 0 < value <= 1 for value in args.classifier_thresholds)
            or not 0 <= args.min_recall <= 1 or not 0 <= args.max_false_alarm_rate <= 1):
        parser.error("counts must be positive; thresholds/rates must be in [0,1]")
    try:
        evaluate(args)
    except (OSError, ValueError, RuntimeError, ImportError, tarfile.TarError) as exc:
        parser.exit(2, f"Temporal evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
