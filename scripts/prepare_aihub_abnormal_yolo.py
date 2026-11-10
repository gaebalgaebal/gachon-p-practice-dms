"""Select AI Hub phone actions and create a scene-deduplicated YOLO dataset."""

import argparse
import csv
import importlib.util
import io
import json
import os
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from dms.aihub_abnormal import category_mapping, split_validation_videos, xywh_to_yolo


MANIFEST_FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
                   "frame_index", "timestamp_s", "task", "label", "split",
                   "source_category", "source_action", "yolo_class",
                   "identity_quality"]


@dataclass
class Selection:
    member: str
    video_id: str
    scene_id: str
    image_name: str
    split: str
    action_label: str
    source_category: str
    source_action: str
    yolo_name: str | None
    yolo_class: int | None
    body_box: list | None
    frame_position: int


def _find_one(source: Path, name: str) -> Path:
    paths = list(source.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {name} below {source}, found {len(paths)}")
    return paths[0]


def scan_label_tar(path: Path, original_split: str, all_normal_frames: bool = False,
                   merge_phone_classes: bool = False) -> list[Selection]:
    selections = []
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not (member.isfile() and member.name.lower().endswith(".json")):
                continue
            payload = json.load(archive.extractfile(member))
            category = (payload.get("scene_info") or {}).get("category_name")
            if category not in {"차량제어", "휴대폰조작", "통화"}:
                continue
            action_label, yolo_name, yolo_class = category_mapping(category)
            video_id = str((payload.get("metadata") or {}).get("video_id", "")).strip()
            scene_id = str((payload.get("scene_info") or {}).get("scene_id", "")).strip()
            frames = (payload.get("scene") or {}).get("data") or []
            if not video_id or not scene_id or not frames:
                raise ValueError(f"Incomplete label metadata: {member.name}")
            selected_frames = list(enumerate(frames)) if all_normal_frames and category == "차량제어" \
                else [(len(frames) // 2, frames[len(frames) // 2])]
            for frame_position, frame in selected_frames:
                occupants = frame.get("occupant") or []
                if len(occupants) != 1:
                    raise ValueError(f"Expected one driver in {scene_id}, found {len(occupants)}")
                image_name = str(frame.get("img_name", "")).strip()
                if not image_name:
                    raise ValueError(f"Missing image name in {scene_id}")
                image_member = f"{video_id}/{scene_id}/img/{image_name}"
                selections.append(Selection(
                    image_member, video_id, scene_id, image_name, original_split,
                    action_label, category, str(occupants[0].get("action", "")),
                    "phone_action" if merge_phone_classes and yolo_class is not None else yolo_name,
                    0 if merge_phone_classes and yolo_class is not None else yolo_class,
                    occupants[0].get("body_b_box") if yolo_class is not None else None,
                    frame_position,
                ))
    return selections


def assign_splits(train: list[Selection], validation: list[Selection], seed: int) -> None:
    assignments = split_validation_videos((item.video_id for item in validation), seed)
    for item in train:
        item.split = "train"
    for item in validation:
        item.split = assignments[item.video_id]


def extract_selected(source_tar: Path, selections: list[Selection], output: Path,
                     manifest_dir: Path) -> tuple[list[dict], list[dict]]:
    by_member = {item.member: item for item in selections}
    if len(by_member) != len(selections):
        raise ValueError("Duplicate selected source image members")
    rows, skipped, seen_members = [], [], set()
    with tarfile.open(source_tar, "r:") as archive:
        for member in archive:
            item = by_member.get(member.name)
            if item is None:
                continue
            seen_members.add(member.name)
            raw = archive.extractfile(member).read()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                image.verify()
            stem = f"{item.scene_id}_{PurePosixPath(item.image_name).stem}"
            image_path = output / "images" / item.split / f"{stem}.jpg"
            label_path = output / "labels" / item.split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if item.yolo_class is None:
                label_text = ""
            else:
                try:
                    x, y, w, h = xywh_to_yolo(item.body_box, width, height)
                except ValueError:
                    skipped.append({"scene_id": item.scene_id, "reason": "invalid_box",
                                    "box": item.body_box, "image_size": [width, height]})
                    continue
                label_text = f"{item.yolo_class} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            image_path.write_bytes(raw)
            label_path.write_text(label_text, encoding="utf-8")
            rows.append({
                "sample_id": f"aihub_abnormal:{stem}", "dataset": "aihub_abnormal",
                "subject_id": f"aihub_abnormal:proxy:{item.video_id}",
                "session_id": f"aihub_abnormal:{item.scene_id}",
                "file_path": Path(os.path.relpath(image_path.resolve(),
                                                   manifest_dir.resolve())).as_posix(),
                "frame_index": item.frame_position, "timestamp_s": 0, "task": "action",
                "label": item.action_label, "split": item.split,
                "source_category": item.source_category,
                "source_action": item.source_action,
                "yolo_class": item.yolo_name or "negative",
                "identity_quality": "video_id_proxy",
            })
            if len(rows) % 5000 == 0:
                print(f"extracted {len(rows)}/{len(selections)} from {source_tar.name}", flush=True)
    for name in sorted(set(by_member) - seen_members):
        item = by_member[name]
        skipped.append({"scene_id": item.scene_id, "reason": "missing_source_image"})
    return rows, skipped


def _validate(path: Path) -> int:
    validator_path = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def build_dataset(source: Path, output: Path, manifest: Path, seed: int = 42,
                  force: bool = False, all_normal_frames: bool = False,
                  merge_phone_classes: bool = False) -> dict:
    if output.exists():
        if not force:
            raise ValueError(f"Output exists; pass --force to replace it: {output}")
        shutil.rmtree(output)
    if manifest.exists() and not force:
        raise ValueError(f"Manifest exists; pass --force to replace it: {manifest}")
    tl4, vl1 = _find_one(source, "TL4.tar"), _find_one(source, "VL1.tar")
    ts4, vs1 = _find_one(source, "TS4.tar"), _find_one(source, "VS1.tar")
    train = scan_label_tar(tl4, "train", all_normal_frames, merge_phone_classes)
    validation = scan_label_tar(vl1, "validation", all_normal_frames, merge_phone_classes)
    assign_splits(train, validation, seed)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    train_rows, train_skipped = extract_selected(ts4, train, output, manifest.parent)
    validation_rows, validation_skipped = extract_selected(
        vs1, validation, output, manifest.parent
    )
    rows = train_rows + validation_rows
    skipped = train_skipped + validation_skipped
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))
    count = _validate(manifest)
    names = "names:\n  0: phone_action\n" if merge_phone_classes else \
        "names:\n  0: phone_use\n  1: phone_call\n"
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        + names,
        encoding="utf-8",
    )
    summary = {
        "samples": count,
        "splits": dict(Counter(row["split"] for row in rows)),
        "actions": dict(Counter(row["label"] for row in rows)),
        "yolo_classes": dict(Counter(row["yolo_class"] for row in rows)),
        "proxy_subjects": len({row["subject_id"] for row in rows}),
        "excluded": dict(Counter(item["reason"] for item in skipped)),
        "exclusion_details": skipped,
        "selection": ("all normal frames; one middle phone frame per scene"
                      if all_normal_frames else "one middle frame per scene"),
        "merge_phone_classes": merge_phone_classes,
        "identity_quality": "video_id_proxy; true driver identity unavailable",
        "box_semantics": "driver body action box, not phone object box",
    }
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/aihub_abnormal_yolo"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/aihub_abnormal_actions.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-normal-frames", action="store_true",
                        help="Use all vehicle-control frames to strengthen negative examples")
    parser.add_argument("--merge-phone-classes", action="store_true",
                        help="Train phone use/call as one phone_action class")
    args = parser.parse_args()
    try:
        summary = build_dataset(args.source, args.output, args.manifest, args.seed, args.force,
                                args.all_normal_frames, args.merge_phone_classes)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
