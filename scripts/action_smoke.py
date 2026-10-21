"""Check the optional training path on random pixels; produces unusable test weights."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/action-smoke'))
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(42)
    manifest = args.output / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'subject_id', 'session_id', 'split', 'label', 'file_path'])
        for split in ('train', 'val', 'test'):
            for label in ('normal', 'phone'):
                for index in range(2):
                    filename = f'{split}_{label}_{index}.png'
                    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(args.output / filename)
                    writer.writerow([filename, f'synthetic_{split}', f'session_{split}', split, label, filename])
    checkpoint = args.output / 'smoke_only.pt'
    subprocess.run([sys.executable, 'scripts/train_action.py', '--manifest', str(manifest),
                    '--output', str(checkpoint), '--epochs', '1', '--batch-size', '2',
                    '--from-scratch', '--smoke-only'], check=True)
    from dms.action import MobileNetAction
    try:
        MobileNetAction(checkpoint)
    except ValueError as exc:
        assert 'smoke/random' in str(exc)
    else:
        raise AssertionError('Random-image smoke checkpoint must not be used for live recognition')
    print('PASS: MobileNet training, checkpoint and test metrics; smoke weights rejected by live adapter.')


if __name__ == '__main__':
    main()
