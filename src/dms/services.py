"""Desktop product services for camera, settings, sessions and diagnostics."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time


APP_NAME = "GachonDMS"


def app_data_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    return Path(root) / APP_NAME if root else Path.home() / ".local" / "share" / APP_NAME


@dataclass
class ProductSettings:
    camera_index: int = 0
    sound_enabled: bool = True
    log_retention_days: int = 30
    output_directory: str = ""
    first_run_complete: bool = False
    experimental_phone_enabled: bool = False

    def validate(self):
        if self.camera_index < 0:
            raise ValueError("camera_index must be nonnegative")
        if not 1 <= self.log_retention_days <= 3650:
            raise ValueError("log_retention_days must be in [1,3650]")
        if not self.output_directory.strip():
            raise ValueError("output_directory must not be empty")


class SettingsService:
    def __init__(self, path: Path | None = None):
        self.path = path or app_data_dir() / "settings.json"

    def load(self) -> ProductSettings:
        if not self.path.exists():
            return ProductSettings(output_directory=str(app_data_dir() / "sessions"))
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        allowed = set(ProductSettings.__dataclass_fields__)
        settings = ProductSettings(**{k: v for k, v in payload.items() if k in allowed})
        settings.validate()
        if not settings.output_directory:
            settings.output_directory = str(app_data_dir() / "sessions")
        return settings

    def save(self, settings: ProductSettings):
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                         dir=self.path.parent, suffix=".tmp") as handle:
            json.dump(asdict(settings), handle, ensure_ascii=False, indent=2)
            temporary = Path(handle.name)
        temporary.replace(self.path)


class CameraService:
    def __init__(self):
        self.capture = None
        self.index = None

    @staticmethod
    def enumerate(max_devices: int = 8) -> list[int]:
        import cv2
        devices = []
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        for index in range(max_devices):
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                devices.append(index)
            capture.release()
        return devices

    def open(self, index: int, width: int = 640, height: int = 480, fps: float = 30):
        import cv2
        self.close()
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened() and platform.system() == "Windows":
            capture.release(); capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError("camera_unavailable")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture, self.index = capture, index

    def read(self):
        if self.capture is None:
            return False, None
        return self.capture.read()

    def reconnect(self) -> bool:
        index = self.index
        if index is None:
            return False
        try:
            self.open(index)
            return True
        except RuntimeError:
            return False

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None


@dataclass
class SessionEvent:
    kind: str
    started_s: float
    ended_s: float
    duration_s: float


@dataclass
class SessionSummary:
    session_id: str
    started_at: str
    ended_at: str = ""
    duration_s: float = 0.0
    camera_index: int = 0
    calibrated: bool = False
    exit_reason: str = "running"
    frames: int = 0
    unknown_seconds: float = 0.0
    coverage: float = 0.0
    fps_mean: float = 0.0
    latency_ms_p95: float = 0.0
    app_version: str = "0.4.0"
    model_hashes: dict[str, str] = field(default_factory=dict)


class SessionService:
    EVENT_FIELDS = ("kind", "started_s", "ended_s", "duration_s")

    def __init__(self, output_root: Path, camera_index: int, model_hashes=None,
                 clock=time.perf_counter):
        self.output_root = Path(output_root)
        self.clock = clock
        now = datetime.now(timezone.utc)
        session_id = now.strftime("%Y%m%dT%H%M%S%fZ")
        self.summary = SessionSummary(session_id, now.isoformat(), camera_index=camera_index,
                                      model_hashes=model_hashes or {})
        self.session_dir = self.output_root / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        (self.session_dir / ".incomplete").write_text(
            json.dumps({"session_id": session_id, "started_at": now.isoformat()}, indent=2),
            encoding="utf-8")
        self.started = clock()
        self.previous_t = None
        self.latencies = []
        self.events = []
        self.active = {}

    def update(self, result, latency_ms: float):
        self.summary.frames += 1
        self.summary.calibrated = self.summary.calibrated or result.calibrated
        self.latencies.append(float(latency_ms))
        if self.previous_t is not None:
            dt = max(0.0, result.timestamp_s - self.previous_t)
            if result.unknown:
                self.summary.unknown_seconds += dt
        self.previous_t = result.timestamp_s
        active_now = {
            "eyes_closed": result.state.drowsiness == "suspected",
            "yawn": result.state.yawn == "yawning",
            "attention_away": result.state.attention == "off_forward",
        }
        for kind, active in active_now.items():
            if active and kind not in self.active:
                self.active[kind] = result.timestamp_s
            elif not active and kind in self.active:
                start = self.active.pop(kind)
                self.events.append(SessionEvent(kind, start, result.timestamp_s,
                                                result.timestamp_s - start))

    @staticmethod
    def _percentile(values, fraction=.95):
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * fraction)]

    def finish(self, exit_reason: str = "user_exit") -> Path:
        end_t = self.previous_t or 0.0
        for kind, start in list(self.active.items()):
            self.events.append(SessionEvent(kind, start, end_t, max(0.0, end_t - start)))
        self.active.clear()
        elapsed = max(0.0, self.clock() - self.started)
        self.summary.ended_at = datetime.now(timezone.utc).isoformat()
        self.summary.duration_s = elapsed
        self.summary.exit_reason = exit_reason
        self.summary.coverage = (max(0.0, 1 - self.summary.unknown_seconds / elapsed)
                                 if elapsed else 0.0)
        self.summary.fps_mean = self.summary.frames / elapsed if elapsed else 0.0
        self.summary.latency_ms_p95 = self._percentile(self.latencies)
        (self.session_dir / "session.json").write_text(
            json.dumps({"summary": asdict(self.summary),
                        "events": [asdict(event) for event in self.events]},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (self.session_dir / "events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.EVENT_FIELDS)
            writer.writeheader(); writer.writerows(asdict(event) for event in self.events)
        (self.session_dir / ".incomplete").unlink(missing_ok=True)
        return self.session_dir


def recover_incomplete_sessions(output_root: Path) -> int:
    root = Path(output_root)
    recovered = 0
    if not root.is_dir():
        return recovered
    for marker in root.glob("*/.incomplete"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"session_id": marker.parent.name, "started_at": "unknown"}
        payload.update({"ended_at": datetime.now(timezone.utc).isoformat(),
                        "exit_reason": "unexpected_shutdown", "recovered": True})
        (marker.parent / "recovered_session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        marker.unlink(missing_ok=True)
        recovered += 1
    return recovered


def cleanup_old_sessions(output_root: Path, retention_days: int, now: float | None = None) -> int:
    if retention_days <= 0:
        raise ValueError("retention_days must be positive")
    root = Path(output_root)
    if not root.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - retention_days * 86400
    removed = 0
    for path in root.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path); removed += 1
    return removed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_manifest(manifest: Path, model_root: Path) -> dict[str, str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected = payload.get("models")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Invalid model manifest")
    verified = {}
    for name, digest in expected.items():
        path = model_root / name
        if not path.is_file():
            raise FileNotFoundError(name)
        actual = file_sha256(path)
        if actual != digest:
            raise ValueError(f"Model integrity check failed: {name}")
        verified[name] = actual
    return verified


def create_diagnostic_report(output: Path, model_paths: list[Path], camera_devices: list[int],
                             recent_error: str = "", performance: dict | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output.parent)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_version": "0.4.0", "platform": platform.platform(),
        "python_runtime": platform.python_version(), "camera_devices": camera_devices,
        "models": [{"name": path.name, "present": path.is_file(),
                    "sha256": file_sha256(path) if path.is_file() else None}
                   for path in model_paths],
        "disk_free_mb": disk.free / 1024 ** 2, "recent_error": recent_error,
        "performance": performance or {},
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return output
