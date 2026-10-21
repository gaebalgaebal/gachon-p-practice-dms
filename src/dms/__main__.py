"""Replay a feature CSV; does not open a camera or run a vision model."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import Features
from .state import Config, StateEngine


def optional_float(value):
    return None if value is None or not value.strip() else float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.config.resolve()}:
        parser.error("Output must differ from input and config")
    try:
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
        engine = StateEngine(config)
        states = []
        with args.input.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            required = {"timestamp_s", "ear", "yaw_degrees", "face_valid"}
            if not required.issubset(rows.fieldnames or []):
                raise ValueError(f"CSV requires: {', '.join(sorted(required))}")
            for line, row in enumerate(rows, start=2):
                try:
                    valid = row["face_valid"]
                    if valid not in {"0", "1"}:
                        raise ValueError("face_valid must be 0 or 1")
                    state = engine.update(Features(
                        float(row["timestamp_s"]), optional_float(row["ear"]),
                        optional_float(row["yaw_degrees"]), valid == "1",
                    ))
                    states.append(json.dumps(asdict(state), allow_nan=False))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"CSV line {line}: {exc}") from exc
        if not states:
            raise ValueError("CSV has no feature rows")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(states) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(states)} rule-based states to {args.output}; not model evaluation.")


if __name__ == "__main__":
    main()
