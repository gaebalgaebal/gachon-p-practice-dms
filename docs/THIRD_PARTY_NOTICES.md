# Third-party notices

The Windows application redistributes runtime components under their respective licenses.

| Component | Version policy | License |
|---|---|---|
| MediaPipe | 0.10.35 | Apache License 2.0 |
| OpenVINO Runtime | 2026.4.1 | Apache License 2.0 |
| OpenCV contrib Python | 4.13.0.92 | Apache License 2.0 |
| NumPy | 2.2.6 | BSD-3-Clause and bundled component licenses |
| PySide6 Essentials | 6.8 or later, below 7 | LGPL-3.0-only or applicable alternative license |
| psutil | 6 or later, below 8 | BSD-3-Clause |

Qt/PySide libraries remain dynamically separated in the `onedir` distribution. A release owner must preserve the license files shipped with each dependency and complete the organization’s license review before external distribution.

Model files have separate distribution terms. The face and eye model terms must be reviewed before bundling, and the optional behavior checkpoint is not part of the base installer.
