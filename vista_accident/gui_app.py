#!/usr/bin/env python3
"""
VISTA — Accident Detection desktop UI.

A native PyQt5 app (not a browser/HTML UI) so there's no local web-server
round trip in the loop: video frames are decoded, run through the pipeline,
and painted straight into a Qt widget in the same process.

Layout:
    - Left:  video preview with an "Upload Video" button and playback
             controls above it.
    - Right: a scrolling report panel. Every confirmed+dispatched alert gets
             a card: severity color strip, kind, timestamp, tracks, which
             channels were notified, and an impact screenshot (the frame at
             the moment the alert fired) you can click to view full-size.

Run:
    python gui_app.py

Requires (in addition to requirements.txt): PyQt5
    pip install PyQt5
"""

import os
import sys
import time
from datetime import datetime

import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QScrollArea, QFrame, QProgressBar, QComboBox,
    QDoubleSpinBox, QSizePolicy, QDialog, QMessageBox,
)

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.detector import Detector
from vista_accident.confirmation import SecondaryConfirmation
from vista_accident.render import SEVERITY_COLORS, draw_overlay

SCREENSHOT_DIR = "vista_screenshots"
ALERT_PANEL_WIDTH = 380


def bgr_to_qpixmap(frame, target_w=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_w:
        pix = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
    return pix


class VideoWorker(QThread):
    """Runs the accident pipeline frame-by-frame on a background thread so
    the UI never freezes while the detector/tracker are working."""

    frame_ready = pyqtSignal(object)          # annotated BGR frame (np.ndarray)
    alert_ready = pyqtSignal(object, str)     # AlertPayload, screenshot_path
    progress = pyqtSignal(int, int)           # frame_idx, total_frames
    finished_processing = pyqtSignal(dict)    # summary stats
    error = pyqtSignal(str)

    def __init__(self, source_path, device="cpu", px_per_meter=None,
                 alert_display_seconds=4.0, camera_id="CAM-01",
                 location="Uploaded Video", parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.device = device
        self.px_per_meter = px_per_meter
        self.alert_display_seconds = alert_display_seconds
        self.camera_id = camera_id
        self.location = location
        self._stop = False
        self._pause = False

    def stop(self):
        self._stop = True

    def toggle_pause(self):
        self._pause = not self._pause
        return self._pause

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                self.error.emit(f"Could not open video: {self.source_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_interval = 1.0 / fps

            os.makedirs(SCREENSHOT_DIR, exist_ok=True)

            pipeline = AccidentPipeline(
                detector=Detector(device=self.device),
                heuristic_cfg=HeuristicConfig(),
                camera_cfg=CameraConfig(camera_id=self.camera_id, location_name=self.location),
                dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
                secondary=SecondaryConfirmation(weights_path=None, device=self.device),
                fps_hint=fps,
            )

            active_alerts = []
            frame_idx = 0

            while not self._stop:
                if self._pause:
                    self.msleep(50)
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                t = frame_idx / fps
                loop_t0 = time.time()

                result = pipeline.process_frame(frame, t)

                for payload in result["alerts"]:
                    # Impact screenshot: the raw frame at the moment the
                    # alert fired, before overlay text is drawn on it.
                    shot_path = os.path.join(
                        SCREENSHOT_DIR, f"{payload.alert_id.replace(':', '_')}.png"
                    )
                    cv2.imwrite(shot_path, frame)
                    active_alerts.append({"payload": payload, "fired_t": t})
                    self.alert_ready.emit(payload, shot_path)

                active_alerts = [
                    a for a in active_alerts
                    if (t - a["fired_t"]) <= self.alert_display_seconds
                ]

                annotated = draw_overlay(frame, result["tracks"], pipeline.history,
                                          active_alerts, t, px_per_meter=self.px_per_meter)
                self.frame_ready.emit(annotated)
                self.progress.emit(frame_idx, total_frames)

                frame_idx += 1

                # Pace roughly to source fps so playback feels live rather
                # than either freezing or racing ahead of the video's clock.
                elapsed = time.time() - loop_t0
                remaining = frame_interval - elapsed
                if remaining > 0:
                    self.msleep(int(remaining * 1000))

            cap.release()

            n_alerts = len(pipeline.confirmed_log)
            n_dispatched = sum(1 for _, _, status in pipeline.confirmed_log if status == "dispatched")
            self.finished_processing.emit({
                "frames": frame_idx, "confirmed": n_alerts, "dispatched": n_dispatched,
            })
        except Exception as e:  # surface errors in the UI instead of a silent thread death
            self.error.emit(str(e))


class AlertCard(QFrame):
    """One entry in the side-panel report: severity strip + details +
    a clickable impact-screenshot thumbnail."""

    def __init__(self, payload, screenshot_path, parent=None):
        super().__init__(parent)
        self.screenshot_path = screenshot_path
        self.setFrameShape(QFrame.StyledPanel)
        color = SEVERITY_COLORS.get(payload.severity, (120, 120, 120))
        hex_color = "#%02x%02x%02x" % (color[2], color[1], color[0])  # BGR -> RGB hex
        self.setStyleSheet(
            f"AlertCard {{ border-left: 6px solid {hex_color}; "
            f"background-color: #262626; border-radius: 4px; margin: 4px; }}"
            f"QLabel {{ color: #eaeaea; }}"
        )

        layout = QHBoxLayout(self)

        thumb = QLabel()
        thumb.setFixedSize(96, 72)
        thumb.setScaledContents(True)
        thumb.setStyleSheet("background-color: black; border-radius: 3px;")
        if screenshot_path and os.path.exists(screenshot_path):
            thumb.setPixmap(QPixmap(screenshot_path))
        thumb.setCursor(Qt.PointingHandCursor)
        thumb.mousePressEvent = self._show_full_image
        layout.addWidget(thumb)

        text_col = QVBoxLayout()
        ts = datetime.now().strftime("%H:%M:%S")
        title = QLabel(f"<b>{payload.kind.upper()}</b>  "
                        f"<span style='color:{hex_color}'>[{payload.severity.upper()}]</span>")
        title.setTextFormat(Qt.RichText)
        subtitle = QLabel(f"t={payload.timestamp:.2f}s  •  tracks={payload.track_ids}  •  {ts}")
        subtitle.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        channels = QLabel("→ " + ", ".join(c.replace("_", " ") for c in payload.channels))
        channels.setStyleSheet("font-size: 11px;")
        confidence = QLabel(f"heuristic streak: {payload.confidence_heuristic} frames"
                             + (f" • secondary: {payload.confidence_secondary:.2f}"
                                if payload.confidence_secondary is not None else ""))
        confidence.setStyleSheet("color: #808080; font-size: 10px;")

        for w in (title, subtitle, channels, confidence):
            w.setWordWrap(True)
            text_col.addWidget(w)
        layout.addLayout(text_col, 1)

    def _show_full_image(self, event):
        if not (self.screenshot_path and os.path.exists(self.screenshot_path)):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Impact frame")
        pix = QPixmap(self.screenshot_path)
        pix = pix.scaledToWidth(900, Qt.SmoothTransformation) if pix.width() > 900 else pix
        lbl = QLabel()
        lbl.setPixmap(pix)
        lay = QVBoxLayout(dlg)
        lay.addWidget(lbl)
        dlg.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISTA — Accident Detection (local)")
        self.resize(1280, 760)
        self.worker = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ---------------- left: video + controls ----------------
        left = QVBoxLayout()

        controls = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.on_upload)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])

        self.calib_spin = QDoubleSpinBox()
        self.calib_spin.setPrefix("px/m: ")
        self.calib_spin.setRange(0.0, 500.0)
        self.calib_spin.setValue(0.0)
        self.calib_spin.setToolTip("Optional calibration (pixels per meter). "
                                    "0 = show raw px/s instead of km/h.")

        controls.addWidget(self.upload_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        controls.addWidget(QLabel("Device:"))
        controls.addWidget(self.device_combo)
        controls.addWidget(self.calib_spin)
        left.addLayout(controls)

        self.video_label = QLabel("Upload a video to begin analysis.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.video_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        left.addWidget(self.progress_bar)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #888;")
        left.addWidget(self.status_label)

        root.addLayout(left, 3)

        # ---------------- right: report panel ----------------
        right = QVBoxLayout()
        header = QLabel("Incident Report")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        right.addWidget(header)

        self.summary_label = QLabel("No alerts yet.")
        self.summary_label.setStyleSheet("color: #888; padding: 0 6px;")
        right.addWidget(self.summary_label)

        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_container = QWidget()
        self.report_layout = QVBoxLayout(self.report_container)
        self.report_layout.addStretch(1)
        self.report_scroll.setWidget(self.report_container)
        right.addWidget(self.report_scroll, 1)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(ALERT_PANEL_WIDTH)
        root.addWidget(right_widget)

        self.alert_count = 0

    # ------------------------------------------------------------
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video to analyze", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if not path:
            return
        self.start_analysis(path)

    def start_analysis(self, path):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        # clear previous report
        while self.report_layout.count() > 1:
            item = self.report_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.alert_count = 0
        self.summary_label.setText("No alerts yet.")

        px_per_meter = self.calib_spin.value() or None
        self.worker = VideoWorker(
            source_path=path,
            device=self.device_combo.currentText(),
            px_per_meter=px_per_meter,
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.alert_ready.connect(self.on_alert)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_processing.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.upload_btn.setEnabled(False)
        self.status_label.setText(f"Analyzing: {os.path.basename(path)}")

    def on_pause(self):
        if not self.worker:
            return
        paused = self.worker.toggle_pause()
        self.pause_btn.setText("Resume" if paused else "Pause")

    def on_stop(self):
        if self.worker:
            self.worker.stop()
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.upload_btn.setEnabled(True)
        self.status_label.setText("Stopped.")

    def on_frame(self, frame):
        pix = bgr_to_qpixmap(frame, target_w=self.video_label.width() or 800)
        self.video_label.setPixmap(pix)

    def on_alert(self, payload, screenshot_path):
        self.alert_count += 1
        card = AlertCard(payload, screenshot_path)
        self.report_layout.insertWidget(self.report_layout.count() - 1, card)
        self.summary_label.setText(f"{self.alert_count} alert(s) dispatched.")

    def on_progress(self, frame_idx, total_frames):
        if total_frames:
            self.progress_bar.setMaximum(total_frames)
            self.progress_bar.setValue(frame_idx)
        else:
            self.progress_bar.setMaximum(0)  # indeterminate

    def on_finished(self, summary):
        self.status_label.setText(
            f"Done — {summary['frames']} frames processed, "
            f"{summary['confirmed']} confirmed, {summary['dispatched']} dispatched."
        )
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.upload_btn.setEnabled(True)

    def on_error(self, message):
        QMessageBox.critical(self, "Analysis error", message)
        self.status_label.setText("Error — see dialog.")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.upload_btn.setEnabled(True)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
