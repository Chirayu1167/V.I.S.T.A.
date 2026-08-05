#!/usr/bin/env python3
"""
VISTA — Accident Detection desktop UI.

A native PyQt5 app (not a browser/HTML UI) so there's no local web-server
round trip in the loop: video frames are decoded, run through the pipeline,
and painted straight into a Qt widget in the same process.

Layout:
    - Left:  video preview with an "Upload Video" button and playback
             controls above it.
    - Right: a scrolling incident report. Each confirmed+dispatched alert
             (above the chosen severity threshold) gets one compact card:
             severity color strip, kind/timestamp/tracks/channels, and a
             3-shot filmstrip (before / impact / after) — click any thumb
             to view it full-size.

Run:
    python gui_app.py

Requires (in addition to requirements.txt): PyQt5
    pip install PyQt5
"""

import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QScrollArea, QFrame, QProgressBar, QComboBox,
    QDoubleSpinBox, QSizePolicy, QDialog, QMessageBox, QGroupBox, QCheckBox,
)

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.detector import Detector
from vista_accident.confirmation import SecondaryConfirmation
from vista_accident.render import SEVERITY_COLORS, SEVERITY_RANK, SpeedEstimator, draw_overlay

SCREENSHOT_DIR = "vista_screenshots"
CLIP_DIR = "vista_clips"          # per-alert video clips (pre/post impact), saved when a pipeline runs
ALERT_PANEL_WIDTH = 400
BEFORE_OFFSET_S = 0.6   # how far back the "before" shot is grabbed from
AFTER_DELAY_S = 0.8     # how long after the alert the "after" shot is grabbed

# Incident grouping (DISPLAY-ONLY — never suppresses a dispatch, only merges
# how the SAME crash is presented). Multiple heuristic kinds of one physical
# crash arrive a few seconds apart at the same spot (collision ->
# speed_drop). These are grouped into a single report card with ONE
# clip + ONE screenshot set so one accident = one card. Groups are tight:
# same spot within INCIDENT_MERGE_S and INCIDENT_MERGE_PX, or shared tracks.
INCIDENT_MERGE_S = 6.0
INCIDENT_MERGE_PX = 200.0

STYLE_SHEET = """
QMainWindow { background-color: #1b1b1f; }
QWidget { color: #e8e8ea; font-size: 12px; }
QGroupBox {
    border: 1px solid #34343a; border-radius: 6px; margin-top: 10px;
    padding: 10px 8px 8px 8px; font-weight: 600; color: #b8b8bd;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background-color: #2c2c33; border: 1px solid #3d3d45; border-radius: 5px;
    padding: 6px 14px;
}
QPushButton:hover { background-color: #35353d; }
QPushButton:disabled { color: #6a6a70; }
QPushButton#uploadBtn { background-color: #3a63d8; border: none; font-weight: 600; }
QPushButton#uploadBtn:hover { background-color: #4670e8; }
QComboBox, QDoubleSpinBox {
    background-color: #26262c; border: 1px solid #3d3d45; border-radius: 4px;
    padding: 3px 6px;
}
QProgressBar {
    border: 1px solid #3d3d45; border-radius: 4px; text-align: center;
    background-color: #26262c; height: 14px;
}
QProgressBar::chunk { background-color: #3a63d8; border-radius: 3px; }
QScrollArea { border: none; }
"""


def bgr_to_qpixmap(frame, target_w=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_w:
        pix = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
    return pix


class VideoWorker(QThread):
    """Runs the accident pipeline frame-by-frame on a background thread so
    the UI never freezes while the detector/tracker are working."""

    frame_ready = pyqtSignal(object)              # annotated BGR frame (np.ndarray)
    alert_ready = pyqtSignal(object, dict)        # AlertPayload, {"before":,"impact":,"after":}
    shot_ready = pyqtSignal(str, str)             # alert_id, after-screenshot path
    clip_ready = pyqtSignal(str, str)             # alert_id, clip path (written a few frames after dispatch)
    progress = pyqtSignal(int, int)               # frame_idx, total_frames
    finished_processing = pyqtSignal(dict)        # summary stats
    error = pyqtSignal(str)

    def __init__(self, source_path, device="cpu", px_per_meter=None,
                 alert_display_seconds=4.0, min_severity="low",
                 camera_id="CAM-01", location="Uploaded Video",
                 clip_dir=CLIP_DIR, run_accident=True, run_violence=False, parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.device = device
        self.px_per_meter = px_per_meter
        self.alert_display_seconds = alert_display_seconds
        self.min_severity = min_severity
        self.camera_id = camera_id
        self.location = location
        self.clip_dir = clip_dir
        self.run_accident = run_accident
        self.run_violence = run_violence
        self._stop = False
        self._pause = False

    def stop(self):
        self._stop = True

    def toggle_pause(self):
        self._pause = not self._pause
        return self._pause

    def _passes_filter(self, severity):
        return SEVERITY_RANK.get(severity, 0) >= SEVERITY_RANK.get(self.min_severity, 0)

    def _save_shot(self, alert_id, phase, frame):
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        safe_id = alert_id.replace(":", "_")
        path = os.path.join(SCREENSHOT_DIR, f"{safe_id}_{phase}.png")
        cv2.imwrite(path, frame)
        return path

    def _lookup_before(self, buffer, target_t):
        """Closest buffered raw frame at/just after target_t (looking back)."""
        best = None
        for bt, bframe in buffer:
            if bt <= target_t:
                best = bframe
            else:
                break
        return best if best is not None else (buffer[0][1] if buffer else None)

    def _nearest_incident(self, payload):
        """Find an existing incident this alert belongs to, or None (new card).

        DISABLED: the earlier display merge folded every alert within
        INCIDENT_MERGE_S into the nearest open incident (with a time-only
        fallback that ignored distance), so real 2nd/3rd accidents at the
        same spot a few seconds later were silently eaten — users saw "1st
        and 3rd crash detected, 2nd and 4th missing". Screenshots/clips of a
        later crash landed on the earlier crash's card. Every dispatch now
        creates its OWN card + screenshot set again (the pipeline fuser
        still collapses sub-event kinds of a single crash).
        """
        return None

    def _find_clip_owner(self, alert_id):
        """Map a sub-alert's clip file to its incident's primary alert id so
        ALL clips of one crash land on the SAME card's Play button."""
        for inc in getattr(self, "_incidents", []):
            if alert_id in inc["clip"]:
                return inc["clip"][0]
        return None

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                self.error.emit(f"Could not open video: {self.source_path}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_interval = 1.0 / fps

            pipeline = None
            if self.run_accident:
                pipeline = AccidentPipeline(
                    detector=Detector(device=self.device),
                    heuristic_cfg=HeuristicConfig(),
                    camera_cfg=CameraConfig(camera_id=self.camera_id, location_name=self.location),
                    dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
                    secondary=SecondaryConfirmation(weights_path=None, device=self.device),
                    fps_hint=fps,
                    clip_dir=self.clip_dir,
                )
            speed_estimator = SpeedEstimator(manual_px_per_meter=self.px_per_meter)

            violence_pipeline = None
            if self.run_violence:
                from vista_accident.violence_pipeline import ViolencePipeline
                violence_pipeline = ViolencePipeline(
                    camera_cfg=CameraConfig(camera_id=self.camera_id, location_name=self.location),
                    dispatch_cfg=DispatchConfig(dashboard_log_path="alerts.jsonl"),
                    device=self.device,
                    fps_hint=fps,
                    clip_dir=self.clip_dir,
                )

            # Rolling buffer of raw (unannotated) frames, just deep enough to
            # look back BEFORE_OFFSET_S for the "before" screenshot.
            buf_len = max(2, int(fps * (BEFORE_OFFSET_S + 0.5)))
            raw_buffer = deque(maxlen=buf_len)

            active_alerts = []
            pending_after = []  # [{"due_t":, "alert_id":}]
            self._incidents = []  # display incidents: [{"t":, "cx":, "cy":, "tracks":set, "ids":[..]}]
            clips_saved = 0
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
                raw_copy = frame.copy()
                raw_buffer.append((t, raw_copy))

                result = {"tracks": [], "alerts": [], "clips_saved": []}
                if pipeline is not None:
                    result = pipeline.process_frame(frame, t)
                if violence_pipeline is not None:
                    vres = violence_pipeline.process_frame(frame, t)
                    result = {
                        **result,
                        "alerts": result["alerts"] + vres["alerts"],
                        "clips_saved": result.get("clips_saved", []) + vres.get("clips_saved", []),
                        "persons": vres.get("persons", []),
                    }
                saved_clips = result.get("clips_saved", [])
                clips_saved += len(saved_clips)
                for vid, cpath in saved_clips:
                    inc = self._find_clip_owner(vid)
                    if inc is not None:
                        vid = inc  # fold sub-alert clips into the incident's card
                    self.clip_ready.emit(vid, cpath)

                for payload in result["alerts"]:
                    if not self._passes_filter(payload.severity):
                        continue
                    inc = {
                        "t": payload.timestamp,
                        "cx": payload.meta.get("cx"),
                        "cy": payload.meta.get("cy"),
                        "tracks": set(payload.track_ids),
                        "clip": [payload.alert_id],
                    }
                    self._incidents.append(inc)
                    before_frame = self._lookup_before(raw_buffer, t - BEFORE_OFFSET_S)
                    shots = {
                        "before": self._save_shot(payload.alert_id, "before", before_frame)
                        if before_frame is not None else None,
                        "impact": self._save_shot(payload.alert_id, "impact", raw_copy),
                        "after": None,
                    }
                    active_alerts.append({"payload": payload, "fired_t": t})
                    pending_after.append({"due": t + AFTER_DELAY_S, "alert_id": payload.alert_id})
                    self.alert_ready.emit(payload, shots)

                if pending_after:
                    still_pending = []
                    for p in pending_after:
                        if t >= p["due"]:
                            after_path = self._save_shot(p["alert_id"], "after", raw_copy)
                            self.shot_ready.emit(p["alert_id"], after_path)
                        else:
                            still_pending.append(p)
                    pending_after = still_pending

                active_alerts = [
                    a for a in active_alerts
                    if (t - a["fired_t"]) <= self.alert_display_seconds
                ]

                annotated = frame.copy()
                if violence_pipeline is not None:
                    from vista_accident.render import draw_skeletons
                    violent_ids = {tid for a in active_alerts
                                   if a["payload"].kind == "violence"
                                   for tid in a["payload"].track_ids}
                    draw_skeletons(annotated, result.get("persons", []), violent_ids=violent_ids)
                if pipeline is not None:
                    draw_overlay(annotated, result["tracks"], pipeline.history,
                                 active_alerts, t, speed_estimator,
                                 show_alert_panel=False)  # side panel covers this
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
            if pipeline is not None:
                pipeline.close()  # flush any alerts still queued for the async log writer
            if violence_pipeline:
                violence_pipeline.close()

            n_alerts = n_dispatched = 0
            if pipeline is not None:
                n_alerts = len(pipeline.confirmed_log)
                n_dispatched = sum(1 for _, _, status in pipeline.confirmed_log
                                   if status == "dispatched")
            n_violence = 0
            if violence_pipeline:
                n_violence = sum(1 for _, _, status in violence_pipeline.confirmed_log
                                 if status == "dispatched")
            self.finished_processing.emit({
                "frames": frame_idx, "confirmed": n_alerts, "dispatched": n_dispatched,
                "violence": n_violence, "clips_saved": clips_saved,
            })
        except Exception as e:  # surface errors in the UI instead of a silent thread death
            import traceback
            tb = traceback.format_exc()
            try:
                with open("gui_error.log", "a") as f:
                    f.write(f"--- {datetime.now().isoformat()} source={self.source_path} ---\n{tb}\n")
            except OSError:
                pass
            self.error.emit(str(e))
            try:
                pipeline.close()
            except NameError:
                pass  # pipeline wasn't constructed yet (e.g. video failed to open)
            try:
                if violence_pipeline:
                    violence_pipeline.close()
            except NameError:
                pass


class Thumb(QLabel):
    """A small clickable screenshot thumbnail."""

    def __init__(self, path=None, size=(84, 60)):
        super().__init__()
        self._path = None
        self.setFixedSize(*size)
        self.setScaledContents(True)
        self.setStyleSheet("background-color: #101012; border-radius: 3px; color: #555;")
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.mousePressEvent = self._open
        self.set_path(path)

    def set_path(self, path):
        self._path = path
        if path and os.path.exists(path):
            self.setPixmap(QPixmap(path))
            self.setText("")
        else:
            self.setText("…")

    def _open(self, event):
        if not (self._path and os.path.exists(self._path)):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Screenshot")
        pix = QPixmap(self._path)
        if pix.width() > 900:
            pix = pix.scaledToWidth(900, Qt.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lay = QVBoxLayout(dlg)
        lay.addWidget(lbl)
        dlg.exec()


class AlertCard(QFrame):
    """One compact entry in the side-panel report: severity strip, a short
    detail line, and a before/impact/after filmstrip."""

    def __init__(self, payload, shots, parent=None):
        super().__init__(parent)
        self.alert_id = payload.alert_id
        color = SEVERITY_COLORS.get(payload.severity, (120, 120, 120))
        hex_color = "#%02x%02x%02x" % (color[2], color[1], color[0])  # BGR -> RGB hex
        self.setStyleSheet(
            f"AlertCard {{ border-left: 4px solid {hex_color}; "
            f"background-color: #242429; border-radius: 4px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        title = QLabel(
            f"<b>{payload.kind.replace('_', ' ').title()}</b> &nbsp; "
            f"<span style='color:{hex_color}; font-weight:600;'>{payload.severity.upper()}</span>"
        )
        title.setTextFormat(Qt.RichText)
        outer.addWidget(title)

        ts = datetime.now().strftime("%H:%M:%S")
        subtitle = QLabel(
            f"t={payload.timestamp:.1f}s  •  tracks {payload.track_ids}  •  {ts}  •  "
            f"→ {', '.join(c.replace('_', ' ') for c in payload.channels)}"
        )
        subtitle.setStyleSheet("color: #97979d; font-size: 10.5px;")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        strip = QHBoxLayout()
        strip.setSpacing(6)
        self.thumb_before = Thumb(shots.get("before"))
        self.thumb_impact = Thumb(shots.get("impact"))
        self.thumb_after = Thumb(shots.get("after"))
        for lbl, thumb in (("before", self.thumb_before), ("impact", self.thumb_impact),
                           ("after", self.thumb_after)):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(thumb)
            cap = QLabel(lbl)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("color: #6f6f75; font-size: 9.5px;")
            col.addWidget(cap)
            strip.addLayout(col)
        strip.addStretch(1)
        outer.addLayout(strip)

        self.clip_btn = QPushButton("Play clip")
        self.clip_btn.setToolTip("Open the saved pre/post-impact video clip for this alert")
        self.clip_btn.setEnabled(False)
        self.clip_btn.clicked.connect(lambda: self._play_clip())
        outer.addWidget(self.clip_btn)

    def set_after_shot(self, path):
        self.thumb_after.set_path(path)

    def set_clip_path(self, path):
        self._clip_path = path
        self.clip_btn.setEnabled(path and os.path.exists(path))
        self.clip_btn.setText("Play clip" if self.clip_btn.isEnabled() else "Clip not found")

    def _play_clip(self):
        cpath = getattr(self, "_clip_path", None)
        if not cpath or not os.path.exists(cpath):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(cpath))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISTA — Accident Detection")
        self.resize(1320, 780)
        self.worker = None
        self.alert_cards = {}
        self.alert_count = 0

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        root.addLayout(self._build_left_panel(), 3)
        root.addWidget(self._build_right_panel())

    # ------------------------------------------------------------
    def _build_left_panel(self):
        left = QVBoxLayout()
        left.setSpacing(10)

        controls_box = QGroupBox("Controls")
        controls = QHBoxLayout(controls_box)
        controls.setSpacing(10)

        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.setObjectName("uploadBtn")
        self.upload_btn.clicked.connect(self.on_upload)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)

        controls.addWidget(self.upload_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)

        controls.addWidget(QLabel("Device"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        controls.addWidget(self.device_combo)

        self.accident_check = QCheckBox("Accident detection (yolo11m)")
        self.accident_check.setChecked(True)
        self.accident_check.setToolTip(
            "Detect collisions/accidents with the yolo11m vehicle branch. "
            "Runs EVERY frame (the heavy model). Untick it and run the "
            "violence branch alone for a smooth, faster violence-only test."
        )
        controls.addWidget(self.accident_check)

        self.violence_check = QCheckBox("Violence detection (pose)")
        self.violence_check.setToolTip(
            "Also run the pose-based violence/road-rage branch (yolo11n-pose, "
            "auto-downloaded on first run): close persons + aggressive limb "
            "motion (or sustained box overlap for distant CCTV fights) route "
            "alerts to the police control room."
        )
        controls.addWidget(self.violence_check)

        controls.addWidget(QLabel("Min severity"))
        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["low", "medium", "high", "critical"])
        self.severity_combo.setCurrentText("low")
        self.severity_combo.setToolTip(
            "Alerts below this severity are treated as noise: no report card, "
            "no screenshots."
        )
        controls.addWidget(self.severity_combo)

        controls.addWidget(QLabel("Calibration"))
        self.calib_spin = QDoubleSpinBox()
        self.calib_spin.setPrefix("px/m: ")
        self.calib_spin.setRange(0.0, 500.0)
        self.calib_spin.setValue(0.0)
        self.calib_spin.setToolTip(
            "Optional: real camera calibration (pixels per meter). "
            "0 = auto-estimate speed from each object's own size."
        )
        controls.addWidget(self.calib_spin)

        left.addWidget(controls_box)

        self.video_label = QLabel("Upload a video to begin analysis.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0e0e10; color: #6a6a70; border-radius: 6px;"
        )
        self.video_label.setMinimumSize(640, 440)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.video_label, 1)

        status_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #8a8a90;")
        status_row.addWidget(self.progress_bar, 1)
        left.addLayout(status_row)
        left.addWidget(self.status_label)

        return left

    def _build_right_panel(self):
        right_widget = QWidget()
        right_widget.setFixedWidth(ALERT_PANEL_WIDTH)
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)

        header = QLabel("Incident Report")
        header.setFont(QFont("", 13, QFont.Bold))
        right.addWidget(header)

        self.summary_label = QLabel("No alerts yet.")
        self.summary_label.setStyleSheet("color: #8a8a90;")
        right.addWidget(self.summary_label)

        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_container = QWidget()
        self.report_layout = QVBoxLayout(self.report_container)
        self.report_layout.setSpacing(8)
        self.report_layout.addStretch(1)
        self.report_scroll.setWidget(self.report_container)
        right.addWidget(self.report_scroll, 1)

        return right_widget

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
        if not (self.accident_check.isChecked() or self.violence_check.isChecked()):
            QMessageBox.warning(
                self, "Nothing selected",
                "Tick at least one detection branch (accident and/or violence) "
                "before uploading — nothing would be analyzed otherwise.")
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        while self.report_layout.count() > 1:
            item = self.report_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.alert_cards.clear()
        self.alert_count = 0
        self.summary_label.setText("No alerts yet.")

        px_per_meter = self.calib_spin.value() or None
        self.worker = VideoWorker(
            source_path=path,
            device=self.device_combo.currentText(),
            px_per_meter=px_per_meter,
            min_severity=self.severity_combo.currentText(),
            run_accident=self.accident_check.isChecked(),
            run_violence=self.violence_check.isChecked(),
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.alert_ready.connect(self.on_alert)
        self.worker.shot_ready.connect(self.on_shot_ready)
        self.worker.clip_ready.connect(self.on_clip_ready)
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

    def on_alert(self, payload, shots):
        self.alert_count += 1
        card = AlertCard(payload, shots)
        self.alert_cards[payload.alert_id] = card
        self.report_layout.insertWidget(0, card)
        self.summary_label.setText(f"{self.alert_count} alert(s) dispatched.")

    def on_shot_ready(self, alert_id, after_path):
        card = self.alert_cards.get(alert_id)
        if card:
            card.set_after_shot(after_path)

    def on_clip_ready(self, alert_id, clip_path):
        card = self.alert_cards.get(alert_id)
        if card:
            card.set_clip_path(clip_path)

    def on_progress(self, frame_idx, total_frames):
        if total_frames:
            self.progress_bar.setMaximum(total_frames)
            self.progress_bar.setValue(frame_idx)
        else:
            self.progress_bar.setMaximum(0)  # indeterminate

    def on_finished(self, summary):
        clips = summary.get("clips_saved", 0)
        self.status_label.setText(
            f"Done — {summary['frames']} frames processed, "
            f"{summary['confirmed']} confirmed, {summary['dispatched']} dispatched, "
            f"{clips} clip(s) saved to {CLIP_DIR}."
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
    app.setStyleSheet(STYLE_SHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
