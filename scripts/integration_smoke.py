"""Real model smoke test using an official portrait fixture, not a DMS benchmark."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/integration-smoke'))
    args = parser.parse_args()
    import cv2
    import numpy as np

    if args.output.exists():
        parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    portrait_url = 'https://storage.googleapis.com/mediapipe-assets/portrait.jpg'
    with urllib.request.urlopen(portrait_url, timeout=30) as response:
        data = response.read()
    portrait = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if portrait is None:
        raise RuntimeError('Official portrait fixture could not be decoded')
    h, w = portrait.shape[:2]
    scale = min(640 / w, 480 / h)
    small = cv2.resize(portrait, (round(w * scale), round(h * scale)))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sh, sw = small.shape[:2]
    frame[(480 - sh) // 2:(480 - sh) // 2 + sh, (640 - sw) // 2:(640 - sw) // 2 + sw] = small
    video = args.output / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 20., (640, 480))
    if not writer.isOpened():
        raise RuntimeError('Cannot write smoke input video')
    try:
        for index in range(120):
            writer.write(np.zeros_like(frame) if 80 <= index < 100 else frame)
    finally:
        writer.release()
    logs = args.output / 'demo'
    subprocess.run([sys.executable, '-m', 'dms.live', '--video', str(video),
                    '--headless', '--record', '--output', str(logs)], check=True)
    rows = [json.loads(line) for line in (logs / 'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 120, f'Expected 120 frames, got {len(rows)}'
    assert any(row['calibrated'] and row['face_valid'] for row in rows[45:80]), 'Calibration never completed'
    assert all(row['drowsiness'] == 'unknown' and row['attention'] == 'unknown'
               for row in rows[85:100]), 'Blank frames did not become unknown'
    assert any(row['face_valid'] and row['drowsiness'] == 'normal' for row in rows[105:]), 'Face did not recover'
    reader = cv2.VideoCapture(str(logs / 'annotated.mp4'))
    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, 65)
        ok, image = reader.read()
        assert ok, 'Annotated recording cannot be decoded'
        cv2.imwrite(str(args.output / 'preview.jpg'), image)
    finally:
        reader.release()
    print('PASS: actual model detection, calibration, face-loss unknown, recovery, video/UI rendering and JSONL')


if __name__ == '__main__':
    main()
