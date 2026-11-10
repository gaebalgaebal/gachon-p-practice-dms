"""Initialize dataset directories and report implementation readiness."""

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "dataset_setup.json"


def inspect_dataset(name, item):
    root = REPO / item["root"]
    counts = {pattern: len(list(root.glob(pattern))) if root.exists() else 0
              for pattern in item["ready_globs"]}
    ready = root.is_dir() and all(count > 0 for count in counts.values()) \
        and sum(counts.values()) >= item["minimum_files"]
    return {
        "dataset": name, "priority": item["priority"], "ready": ready,
        "root": str(root), "access": item["access"], "matches": counts,
        "official_url": item["official_url"], "next_action": item["next_action"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true",
                        help="create ignored raw/processed/manifest/license directories")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if args.initialize:
        for item in config.values():
            (REPO / item["root"]).mkdir(parents=True, exist_ok=True)
        for relative in ("data/processed", "data/manifests", "data/licenses", "artifacts"):
            (REPO / relative).mkdir(parents=True, exist_ok=True)
    results = sorted((inspect_dataset(name, item) for name, item in config.items()),
                     key=lambda row: row["priority"])
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for row in results:
        mark = "READY" if row["ready"] else "WAIT"
        found = ", ".join(f"{pattern}={count}" for pattern, count in row["matches"].items())
        print(f"[{mark}] {row['priority']}. {row['dataset']} ({row['access']})")
        print(f"  path: {row['root']}")
        print(f"  files: {found}")
        if not row["ready"]:
            print(f"  next: {row['next_action']}")
            print(f"  url: {row['official_url']}")


if __name__ == "__main__":
    main()
