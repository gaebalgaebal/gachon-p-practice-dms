"""Windows desktop application for real-time driver monitoring."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QLineEdit, QProgressBar, QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

from .product import DmsEngine
from .services import (CameraService, ProductSettings, SessionService, SettingsService,
                       app_data_dir, cleanup_old_sessions, create_diagnostic_report,
                       file_sha256, recover_incomplete_sessions, verify_model_manifest)
from .state import Config


APP_TITLE = "운전자 상태 모니터링"


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / relative


def human_message(code: str) -> str:
    return {
        "ready": "보정 완료",
        "hold_forward": "정면을 보고 눈을 자연스럽게 떠주세요.",
        "face_not_found": "카메라에 얼굴이 보이도록 위치를 조정하세요.",
        "multiple_faces": "한 사람만 카메라에 보이게 해주세요.",
        "move_closer": "카메라에 조금 더 가까이 앉아주세요.",
        "look_forward": "고개를 정면으로 유지해주세요.",
        "camera_unavailable": "카메라를 사용할 수 없습니다. 연결과 권한을 확인하세요.",
        "camera_disconnected": "카메라 연결이 끊겼습니다. 다시 연결하고 있습니다.",
    }.get(code, code)


class MonitorWorker(QThread):
    frame_ready = Signal(object)
    result_ready = Signal(object, float)
    failed = Signal(str)
    session_finished = Signal(str, object)

    def __init__(self, camera_index: int, output_root: Path, experimental_phone=False,
                 parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.output_root = output_root
        self.experimental_phone = experimental_phone
        self._stop = False
        self._recalibrate = False
        self.camera = CameraService()

    def request_stop(self):
        self._stop = True

    def request_recalibration(self):
        self._recalibrate = True

    def _models(self):
        face = resource_path("models/face_landmarker.task")
        eye = resource_path("models/open_closed_eye.onnx")
        action = resource_path("models/aihub_abnormal_yolo11n.pt")
        verify_model_manifest(resource_path("configs/model_manifest.json"), face.parent)
        return face, eye, action

    def run(self):
        engine = session = None
        exit_reason = "error"
        try:
            face, eye, action = self._models()
            hashes = {face.name: file_sha256(face), eye.name: file_sha256(eye)}
            action_path = action if self.experimental_phone and action.is_file() else None
            if action_path:
                hashes[action.name] = file_sha256(action)
            engine = DmsEngine.from_models(
                face, Config(), eye, "CPU", action_path, "yolo11", "cpu", .75,
                self.experimental_phone, 5,
            )
            self.camera.open(self.camera_index)
            session = SessionService(self.output_root, self.camera_index, hashes)
            origin = time.perf_counter()
            reconnect_attempts = 0
            while not self._stop:
                ok, frame = self.camera.read()
                if not ok:
                    reconnect_attempts += 1
                    self.failed.emit("camera_disconnected")
                    if reconnect_attempts >= 30:
                        raise RuntimeError("camera_unavailable")
                    if not self.camera.reconnect():
                        time.sleep(.4)
                    continue
                reconnect_attempts = 0
                timestamp = time.perf_counter() - origin
                started = time.perf_counter()
                if self._recalibrate:
                    engine.reset_calibration(); self._recalibrate = False
                result = engine.process(frame, timestamp)
                latency_ms = (time.perf_counter() - started) * 1000
                session.update(result, latency_ms)
                self.frame_ready.emit(frame.copy())
                self.result_ready.emit(result, latency_ms)
            exit_reason = "user_exit"
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.camera.close()
            if engine:
                engine.close()
            if session:
                try:
                    directory = session.finish(exit_reason)
                    self.session_finished.emit(str(directory), asdict(session.summary))
                except Exception as exc:
                    self.failed.emit(f"세션 결과를 저장하지 못했습니다: {exc}")


class StatusCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        self.title = QLabel(title); self.title.setStyleSheet("font-weight: 600")
        self.value = QLabel("대기"); self.value.setStyleSheet("font-size: 22px")
        layout.addWidget(self.title); layout.addWidget(self.value)

    def set_status(self, text: str, level: str):
        colors = {"normal": "#137333", "warning": "#b3261e", "unknown": "#6b7280"}
        self.value.setText(text)
        self.value.setStyleSheet(f"font-size:22px;font-weight:700;color:{colors[level]}")


class SettingsDialog(QDialog):
    def __init__(self, settings: ProductSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        form = QFormLayout(self)
        self.sound = QCheckBox("경고음 사용"); self.sound.setChecked(settings.sound_enabled)
        self.retention = QSpinBox(); self.retention.setRange(1, 3650)
        self.retention.setValue(settings.log_retention_days); self.retention.setSuffix(" 일")
        self.experimental = QCheckBox("휴대폰 시험 기능 사용")
        self.experimental.setChecked(settings.experimental_phone_enabled)
        self.experimental.setToolTip("시험 기능이며 운영 안전 판정에 포함되지 않습니다.")
        available = resource_path("models/aihub_abnormal_yolo11n.pt").is_file()
        self.experimental.setEnabled(available)
        if not available:
            self.experimental.setChecked(False)
            self.experimental.setToolTip("시험 기능 확장 패키지가 설치되지 않았습니다.")
        self.output = QLineEdit(settings.output_directory)
        browse = QPushButton("찾기")
        browse.clicked.connect(self.choose_output)
        output_row = QHBoxLayout(); output_row.addWidget(self.output); output_row.addWidget(browse)
        form.addRow(self.sound); form.addRow("로그 보관 기간", self.retention)
        form.addRow(self.experimental)
        form.addRow("결과 저장 폴더", output_row)
        clear = QPushButton("저장된 세션 로그 삭제")
        if parent is not None and hasattr(parent, "clear_logs"):
            clear.clicked.connect(parent.clear_logs)
        form.addRow(clear)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply(self, settings: ProductSettings):
        settings.sound_enabled = self.sound.isChecked()
        settings.log_retention_days = self.retention.value()
        settings.experimental_phone_enabled = self.experimental.isChecked()
        settings.output_directory = self.output.text().strip()

    def choose_output(self):
        selected = QFileDialog.getExistingDirectory(self, "결과 저장 폴더", self.output.text())
        if selected:
            self.output.setText(selected)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 760)
        self.settings_service = SettingsService()
        try:
            self.settings = self.settings_service.load()
        except (OSError, ValueError, json.JSONDecodeError):
            self.settings = ProductSettings(output_directory=str(app_data_dir() / "sessions"))
        if not resource_path("models/aihub_abnormal_yolo11n.pt").is_file():
            self.settings.experimental_phone_enabled = False
        recover_incomplete_sessions(Path(self.settings.output_directory))
        cleanup_old_sessions(Path(self.settings.output_directory), self.settings.log_retention_days)
        self.worker = None
        self.session_path = None
        self.last_summary = {}
        self.last_result = None
        self.last_beep = -10.0
        self.stack = QStackedWidget(); self.setCentralWidget(self.stack)
        self._build_start(); self._build_calibration(); self._build_monitor(); self._build_result()
        self.stack.setCurrentWidget(self.start_page)

    def _page(self, title, subtitle):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(32, 24, 32, 24)
        heading = QLabel(title); heading.setStyleSheet("font-size:28px;font-weight:700")
        detail = QLabel(subtitle); detail.setWordWrap(True); detail.setStyleSheet("color:#4b5563")
        layout.addWidget(heading); layout.addWidget(detail)
        return page, layout

    def _build_start(self):
        self.start_page, layout = self._page("운전자 상태 모니터링",
            "카메라를 선택한 뒤 시작하세요. 영상은 기본적으로 저장되지 않습니다.")
        self.camera_combo = QComboBox(); layout.addWidget(self.camera_combo)
        guide = QLabel("1. 카메라 선택  ·  2. 정면 보정  ·  3. 실시간 모니터링")
        guide.setStyleSheet("padding:12px;background:#eef4ff;border-radius:6px")
        layout.addWidget(guide)
        self.check_label = QLabel("장치 점검 전"); layout.addWidget(self.check_label)
        row = QHBoxLayout()
        refresh = QPushButton("카메라 찾기"); refresh.clicked.connect(self.refresh_cameras)
        settings = QPushButton("설정"); settings.clicked.connect(self.open_settings)
        diagnostic = QPushButton("진단 리포트"); diagnostic.clicked.connect(self.export_diagnostic)
        row.addWidget(refresh); row.addWidget(settings); row.addWidget(diagnostic); row.addStretch()
        layout.addLayout(row)
        self.start_button = QPushButton("모니터링 시작")
        self.start_button.setMinimumHeight(48); self.start_button.clicked.connect(self.start_monitoring)
        layout.addStretch(); layout.addWidget(self.start_button)
        self.stack.addWidget(self.start_page)
        self.refresh_cameras()

    def _build_calibration(self):
        self.calibration_page, layout = self._page("사용자 보정",
            "카메라를 정면으로 보고 눈을 자연스럽게 떠주세요.")
        self.calibration_video = QLabel("카메라 준비 중")
        self.calibration_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.calibration_video.setMinimumSize(640, 480)
        self.calibration_progress = QProgressBar(); self.calibration_progress.setRange(0, 100)
        self.calibration_help = QLabel("얼굴을 화면 중앙에 맞춰주세요.")
        layout.addWidget(self.calibration_video, 1); layout.addWidget(self.calibration_progress)
        layout.addWidget(self.calibration_help)
        cancel = QPushButton("취소"); cancel.clicked.connect(self.stop_monitoring)
        layout.addWidget(cancel)
        self.stack.addWidget(self.calibration_page)

    def _build_monitor(self):
        self.monitor_page, layout = self._page("실시간 모니터링",
            "위험 상태가 지속되면 화면과 경고음으로 알립니다.")
        content = QHBoxLayout(); self.video = QLabel(); self.video.setMinimumSize(640, 480)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter); content.addWidget(self.video, 3)
        side = QVBoxLayout()
        self.overall = QLabel("● 정상"); self.overall.setStyleSheet("font-size:26px;color:#137333")
        side.addWidget(self.overall)
        self.eye_card = StatusCard("눈 상태"); self.yawn_card = StatusCard("하품")
        self.attention_card = StatusCard("주시 방향")
        side.addWidget(self.eye_card); side.addWidget(self.yawn_card); side.addWidget(self.attention_card)
        self.phone_card = StatusCard("휴대폰 시험 기능"); self.phone_card.setVisible(False)
        side.addWidget(self.phone_card)
        self.performance = QLabel("FPS 계산 중"); side.addWidget(self.performance); side.addStretch()
        content.addLayout(side, 2); layout.addLayout(content, 1)
        controls = QHBoxLayout()
        self.mute = QPushButton("경고음 끄기"); self.mute.setCheckable(True)
        self.mute.clicked.connect(self.toggle_mute)
        recalibrate = QPushButton("재보정"); recalibrate.clicked.connect(self.recalibrate)
        stop = QPushButton("세션 종료"); stop.clicked.connect(self.stop_monitoring)
        controls.addWidget(self.mute); controls.addWidget(recalibrate); controls.addStretch()
        controls.addWidget(stop); layout.addLayout(controls)
        self.stack.addWidget(self.monitor_page)

    def _build_result(self):
        self.result_page, layout = self._page("세션 결과", "영상 없이 이벤트와 성능 요약만 저장했습니다.")
        self.result_label = QLabel(); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size:18px"); layout.addWidget(self.result_label)
        row = QHBoxLayout()
        export = QPushButton("결과 폴더 선택 내보내기"); export.clicked.connect(self.export_result)
        restart = QPushButton("새 세션"); restart.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.start_page))
        row.addWidget(export); row.addWidget(restart); row.addStretch(); layout.addLayout(row)
        layout.addStretch(); self.stack.addWidget(self.result_page)

    @Slot()
    def refresh_cameras(self):
        self.camera_combo.clear()
        try:
            devices = CameraService.enumerate()
        except Exception:
            devices = []
        for index in devices:
            self.camera_combo.addItem(f"카메라 {index}", index)
        model_ok = False
        try:
            verify_model_manifest(resource_path("configs/model_manifest.json"),
                                  resource_path("models"))
            model_ok = True
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        output = Path(self.settings.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        disk_ok = shutil.disk_usage(output).free >= 1024 ** 3
        self.start_button.setEnabled(bool(devices) and model_ok and disk_ok)
        if not devices:
            status = "사용 가능한 카메라를 찾지 못했습니다."
        elif not model_ok:
            status = "필수 모델이 없거나 손상되었습니다."
        elif not disk_ok:
            status = "저장공간이 1GB 미만입니다."
        else:
            status = "카메라·모델·저장공간 점검 완료"
        self.check_label.setText(status)
        preferred = self.camera_combo.findData(self.settings.camera_index)
        if preferred >= 0:
            self.camera_combo.setCurrentIndex(preferred)

    @Slot()
    def start_monitoring(self):
        index = self.camera_combo.currentData()
        if index is None:
            return
        self.settings.camera_index = int(index); self.settings.first_run_complete = True
        self.settings_service.save(self.settings)
        self.calibration_progress.setValue(0)
        self.stack.setCurrentWidget(self.calibration_page)
        self.worker = MonitorWorker(index, Path(self.settings.output_directory),
                                    self.settings.experimental_phone_enabled, self)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.result_ready.connect(self.show_result)
        self.worker.failed.connect(self.show_error)
        self.worker.session_finished.connect(self.session_complete)
        self.phone_card.setVisible(self.settings.experimental_phone_enabled)
        self.worker.start()

    @Slot(object)
    def show_frame(self, frame):
        rgb = frame[:, :, ::-1].copy()
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                       QImage.Format.Format_RGB888).copy()
        target = self.calibration_video if self.stack.currentWidget() == self.calibration_page else self.video
        target.setPixmap(QPixmap.fromImage(image).scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))

    @Slot(object, float)
    def show_result(self, result, latency_ms):
        self.last_result = result
        if not result.calibrated:
            self.calibration_progress.setValue(round(result.calibration_progress * 100))
            self.calibration_help.setText(human_message(result.calibration_message))
            return
        if self.stack.currentWidget() == self.calibration_page:
            self.stack.setCurrentWidget(self.monitor_page)
        state = result.state
        self.eye_card.set_status("위험 · 장시간 감김" if state.drowsiness == "suspected" else
                                 "정상" if state.drowsiness == "normal" else "판단 대기",
                                 "warning" if state.drowsiness == "suspected" else
                                 "normal" if state.drowsiness == "normal" else "unknown")
        self.yawn_card.set_status("주의 · 하품" if state.yawn == "yawning" else
                                  "정상" if state.yawn == "normal" else "판단 대기",
                                  "warning" if state.yawn == "yawning" else
                                  "normal" if state.yawn == "normal" else "unknown")
        self.attention_card.set_status("주의 · 전방 이탈" if state.attention == "off_forward" else
                                       "정상" if state.attention == "forward" else "판단 대기",
                                       "warning" if state.attention == "off_forward" else
                                       "normal" if state.attention == "forward" else "unknown")
        if self.settings.experimental_phone_enabled:
            self.phone_card.set_status("시험 감지" if state.action == "phone" else "미감지",
                                       "warning" if state.action == "phone" else "normal")
        if result.warning:
            self.overall.setText("▲ 운전자 상태 확인 필요")
            self.overall.setStyleSheet("font-size:26px;font-weight:700;color:#b3261e")
            if self.settings.sound_enabled and not self.mute.isChecked() and time.monotonic() - self.last_beep > 3:
                QApplication.beep(); self.last_beep = time.monotonic()
        else:
            self.overall.setText("● 모니터링 중")
            self.overall.setStyleSheet("font-size:26px;font-weight:700;color:#137333")
        fps = 1000 / latency_ms if latency_ms > 0 else 0
        self.performance.setText(f"추론 {fps:.1f} FPS · {latency_ms:.1f} ms")

    @Slot(str)
    def show_error(self, message):
        if message == "camera_disconnected":
            self.overall.setText("카메라 재연결 중")
            return
        QMessageBox.critical(self, "실행 오류", human_message(message))

    @Slot()
    def stop_monitoring(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop(); self.worker.wait(5000)
        else:
            self.stack.setCurrentWidget(self.start_page)

    @Slot(str, object)
    def session_complete(self, path, summary):
        self.session_path = Path(path); self.last_summary = summary
        self.result_label.setText(
            f"운행 시간: {summary['duration_s']:.1f}초\n"
            f"처리 프레임: {summary['frames']}\n"
            f"상태 판정 coverage: {summary['coverage'] * 100:.1f}%\n"
            f"평균 처리 FPS: {summary['fps_mean']:.1f}\n"
            f"p95 지연: {summary['latency_ms_p95']:.1f}ms\n"
            f"종료 사유: {summary['exit_reason']}"
        )
        self.stack.setCurrentWidget(self.result_page)

    def recalibrate(self):
        if self.worker:
            self.worker.request_recalibration(); self.calibration_progress.setValue(0)
            self.stack.setCurrentWidget(self.calibration_page)

    def toggle_mute(self, checked):
        self.mute.setText("경고음 켜기" if checked else "경고음 끄기")

    def open_settings(self):
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.experimental.isChecked() and not self.settings.experimental_phone_enabled:
                QMessageBox.warning(self, "시험 기능",
                    "휴대폰 감지는 시험 기능이며 운영 안전 판정에 포함되지 않습니다.")
            dialog.apply(self.settings); self.settings_service.save(self.settings)
            cleanup_old_sessions(Path(self.settings.output_directory),
                                 self.settings.log_retention_days)

    def clear_logs(self):
        answer = QMessageBox.question(self, "세션 로그 삭제",
                                      "저장된 세션 로그를 모두 삭제하시겠습니까?")
        if answer == QMessageBox.StandardButton.Yes:
            root = Path(self.settings.output_directory)
            if root.is_dir():
                for path in root.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
            QMessageBox.information(self, "세션 로그 삭제", "세션 로그를 삭제했습니다.")

    def export_result(self):
        if not self.session_path:
            return
        destination = QFileDialog.getExistingDirectory(self, "결과 내보내기")
        if destination:
            import shutil
            target = Path(destination) / self.session_path.name
            if target.exists():
                QMessageBox.warning(self, "내보내기", "같은 이름의 결과 폴더가 이미 있습니다.")
                return
            shutil.copytree(self.session_path, target)
            QMessageBox.information(self, "내보내기", "결과를 내보냈습니다.")

    def export_diagnostic(self):
        destination, _ = QFileDialog.getSaveFileName(self, "진단 리포트 저장", "diagnostic.json",
                                                     "JSON (*.json)")
        if destination:
            models = [resource_path("models/face_landmarker.task"),
                      resource_path("models/open_closed_eye.onnx")]
            create_diagnostic_report(Path(destination), models,
                                     [self.camera_combo.itemData(i) for i in range(self.camera_combo.count())],
                                     performance=self.last_summary)
            QMessageBox.information(self, "진단 리포트", "진단 리포트를 저장했습니다.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop(); self.worker.wait(5000)
        event.accept()


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("Gachon DMS Team")
    app.setStyleSheet("""
        QWidget { font-family: 'Malgun Gothic'; font-size: 14px; }
        QPushButton { padding: 8px 14px; }
        QProgressBar { min-height: 24px; text-align: center; }
    """)
    window = MainWindow(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
