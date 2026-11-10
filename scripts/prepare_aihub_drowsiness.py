"""Build a subject-disjoint DMS manifest directly from AI Hub ZIP archives."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from dms.aihub_drowsiness import (
    controlled_labels, environment_from_archive, eye_label,
    session_from_filename, subject_from_member,
)
from dms.mrl_eye import split_subjects


FIELDS = ["sample_id", "dataset", "subject_id", "session_id", "file_path",
          "frame_index", "timestamp_s", "task", "label", "split",
          "image_member", "label_archive", "label_member", "storage",
          "source_partition", "environment", "gender", "age", "mask",
          "glasses", "cap", "metadata_filename", "metadata_filename_match",
          "image_match_method"]


def validate_dms_manifest(path: Path) -> int:
    validator_path = Path(__file__).with_name("validate_manifest.py")
    spec = importlib.util.spec_from_file_location("dms_manifest_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(path, "dms")


def find_pairs(root: Path):
    pairs = []
    for label_archive in sorted(root.rglob("[[]라벨]*.zip")):
        environment = environment_from_archive(label_archive.name)
        if environment == "semi_controlled":
            continue
        source_archive = label_archive.with_name(label_archive.name.replace("[라벨]", "[원천]"))
        if not source_archive.is_file():
            raise ValueError(f"Missing source archive for {label_archive}")
        partition = "train_source" if "Training" in label_archive.parts else "validation_source"
        pairs.append((source_archive, label_archive, environment, partition))
    if not pairs:
        raise ValueError(f"No bbox source/label archive pairs found below {root}")
    return pairs


def collect_subjects(pairs) -> set[str]:
    subjects = set()
    for _, label_archive, environment, _ in pairs:
        with ZipFile(label_archive) as labels:
            for member in labels.namelist():
                if member.lower().endswith(".json"):
                    subjects.add(subject_from_member(member, environment))
    return subjects


def real_road_key(filename: str):
    fields = PurePosixPath(filename).stem.split("_")
    if len(fields) != 9 or fields[0] != "R":
        return None
    return tuple(fields[:5] + fields[-1:])


def accessory_codes(filename: str):
    fields = PurePosixPath(filename).stem.split("_")
    return tuple(fields[5:8]) if len(fields) == 9 else ()


def member_index(archive: ZipFile, environment: str):
    exact, canonical = {}, {}
    for member in archive.namelist():
        if not member.lower().endswith(".jpg"):
            continue
        path = PurePosixPath(member)
        key = (str(path.parent), path.name)
        if key in exact:
            raise ValueError(f"Duplicate source member key: {key}")
        exact[key] = member
        if environment == "real_road":
            broad_key = real_road_key(path.name)
            if broad_key is not None:
                canonical.setdefault((str(path.parent), broad_key), []).append(member)
    return exact, canonical


def choose_real_road_member(candidates, member_filename, metadata_filename):
    member_codes = accessory_codes(member_filename)
    metadata_codes = accessory_codes(metadata_filename)
    def distance(candidate, target):
        codes = accessory_codes(PurePosixPath(candidate).name)
        return sum(left != right for left, right in zip(codes, target)) if target else 99
    return min(candidates, key=lambda candidate: (
        distance(candidate, member_codes), distance(candidate, metadata_codes), candidate))


def iter_task_rows(source_path, label_path, environment, partition, assignments,
                   output_parent, stats):
    with ZipFile(source_path) as sources, ZipFile(label_path) as labels:
        images, canonical_images = member_index(sources, environment)
        relative_source = Path(os.path.relpath(source_path.resolve(), output_parent.resolve())).as_posix()
        relative_label = Path(os.path.relpath(label_path.resolve(), output_parent.resolve())).as_posix()
        for label_member in labels.namelist():
            if not label_member.lower().endswith(".json"):
                continue
            stats["label_json"] += 1
            payload = json.loads(labels.read(label_member))
            filename = str(payload.get("FileInfo", {}).get("FileName", ""))
            if not filename:
                raise ValueError(f"Missing FileInfo.FileName: {label_member}")
            parent = str(PurePosixPath(label_member).parent)
            member_filename = PurePosixPath(label_member).with_suffix(".jpg").name
            image_member = images.get((parent, member_filename))
            match_method = "label_member"
            if image_member is None:
                image_member = images.get((parent, filename))
                match_method = "metadata_filename"
            if image_member is None and environment == "real_road":
                broad_key = real_road_key(member_filename) or real_road_key(filename)
                candidates = canonical_images.get((parent, broad_key), [])
                if candidates:
                    image_member = choose_real_road_member(
                        candidates, member_filename, filename)
                    match_method = "real_road_accessory_fallback"
            if image_member is None:
                stats["unmatched_label"] += 1
                continue
            if match_method == "real_road_accessory_fallback":
                stats["accessory_fallback"] += 1
            actual_filename = PurePosixPath(image_member).name
            subject = subject_from_member(label_member, environment)
            common = {
                "dataset": "aihub_drowsiness", "subject_id": subject,
                "session_id": session_from_filename(subject, actual_filename),
                "file_path": relative_source, "frame_index": 0, "timestamp_s": 0,
                "split": assignments[subject], "image_member": image_member,
                "label_archive": relative_label, "label_member": label_member,
                "storage": "zip", "source_partition": partition,
                "environment": environment,
                "gender": payload.get("UserInfo", {}).get("Gender", ""),
                "age": payload.get("UserInfo", {}).get("Age", ""),
                "mask": payload.get("Accessory", {}).get("Mask", ""),
                "glasses": payload.get("Accessory", {}).get("Glasses", ""),
                "cap": payload.get("Accessory", {}).get("Cap", ""),
                "metadata_filename": filename,
                "metadata_filename_match": filename == actual_filename,
                "image_match_method": match_method,
            }
            labels_for_image = []
            eye = eye_label(payload)
            if eye is not None:
                labels_for_image.append(("eye_state", eye))
            if environment == "controlled":
                drowsiness, direction = controlled_labels(filename)
                if drowsiness:
                    labels_for_image.append(("drowsiness", drowsiness))
                    labels_for_image.append(("head_pose", direction))
            for task, label in labels_for_image:
                stats[f"task_{task}"] += 1
                yield {**common, "sample_id": f"aihub_drowsiness:{partition}:{PurePosixPath(image_member).stem}:{task}",
                       "task": task, "label": label}


def build_manifest(root: Path, output: Path, train_ratio: float = .70,
                   val_ratio: float = .15, seed: int = 42, force: bool = False) -> int:
    if not root.is_dir():
        raise ValueError(f"AI Hub drowsiness directory not found: {root}")
    if output.exists() and not force:
        raise ValueError(f"Output exists; pass --force to replace it: {output}")
    pairs = find_pairs(root)
    subjects = collect_subjects(pairs)
    assignments = split_subjects(subjects, train_ratio, val_ratio, seed)
    stats = {"label_json": 0, "unmatched_label": 0, "accessory_fallback": 0,
             "task_eye_state": 0, "task_drowsiness": 0, "task_head_pose": 0,
             "subjects": len(subjects)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerows(iter_task_rows(*pair, assignments, output.parent, stats))
    count = validate_dms_manifest(output)
    summary = {**stats, "manifest_rows": count, "seed": seed,
               "train_ratio": train_ratio, "val_ratio": val_ratio}
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("data/manifests/aihub_drowsiness.csv"))
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        count = build_manifest(args.source, args.output, args.train_ratio,
                               args.val_ratio, args.seed, args.force)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"PASS: wrote {count} AI Hub drowsiness task samples to {args.output}")


if __name__ == "__main__":
    main()
