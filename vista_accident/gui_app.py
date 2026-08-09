"""
VISTA — Accident Detection desktop UI.

A native PyQt5 app (not a browser/HTML UI) so there's no local web-server
round trip in the loop: video frames are decoded, run through the pipeline,
and painted straight into a Qt widget in the same process.

Layout:
    - Header:   VISTA wordmark + section label, live status pill, and the
                primary "Upload video" action.
    - Toolbar:  playback + detection controls, then a second row of
                severity/device/profile dropdowns and calibration.
    - Left:     video preview card with an empty state until a video is
                loaded, then live annotated frames + analysis progress.
    - Right:    "Incident Report" card. Each confirmed+dispatched alert
                (above the chosen severity threshold) gets one compact
                card: severity color strip, kind/timestamp/tracks/channels,
                and a 3-shot filmstrip (before / impact / after) — click
                any thumb to view it full-size.
    - Footer:   full-width status strip with the detailed status message.

Run:
    python gui_app.py

Requires (in addition to requirements.txt): PyQt5
    pip install PyQt5
"""

import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
# IMPORTANT: PyTorch must load BEFORE PyQt5
import torch
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal, QSize, QPointF, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QDesktopServices, QColor, QPainter, QPainterPath,
    QPen, QBrush, QIcon,
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QScrollArea, QFrame, QProgressBar, QComboBox,
    QCheckBox, QDoubleSpinBox, QSizePolicy, QDialog, QMessageBox, QGroupBox,
    QLineEdit, QSpinBox, QGraphicsDropShadowEffect, QStackedLayout, QSlider,
)

from vista_accident import AccidentPipeline, CameraConfig, DispatchConfig, HeuristicConfig
from vista_accident.camera_profile import find_profiles, load_profile, save_profile
from vista_accident.detector import Detector
from vista_accident.confirmation import SecondaryConfirmation
from vista_accident.render import SEVERITY_COLORS, SEVERITY_RANK, SpeedEstimator, draw_overlay
from vista_accident.tools.calibrate_camera import build_homography, render_birdseye_preview

# Absolute base dir of THIS file (not cwd!): the GUI can be launched from
# anywhere (shortcut, Explorer, terminal) — the alert log, clips, screenshots
# and the control-room console must all anchor to the same place, otherwise
# the console watches a different alerts.jsonl than the GUI writes and shows
# nothing.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERTS_LOG = os.path.join(BASE_DIR, "alerts.jsonl")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "vista_screenshots")
CLIP_DIR = os.path.join(BASE_DIR, "vista_clips")  # per-alert clips (pre/post impact)
RECIPIENTS_PATH = os.path.join(BASE_DIR, "recipients.json")
ACKS_PATH = os.path.join(BASE_DIR, "acks.jsonl")
CONTROL_ROOM_URL = "http://localhost:8787"
ALERT_PANEL_WIDTH = 420
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

# ---------------------------------------------------------------------------
# VISUAL THEME (UI/presentation only — no functional code below this point
# was changed to produce this theme).
#
# Palette matches the VISTA reference spec: light background, white cards,
# dark-navy primary text, blue-gray secondary text, soft borders, generous
# radii/padding and 40-48px control heights.
# ---------------------------------------------------------------------------
COLOR_BG = "#F5F7FA"
COLOR_BG_RAISED = "#FFFFFF"
COLOR_CARD = "#FFFFFF"
COLOR_CARD_2 = "#EEF1F6"
COLOR_BORDER = "#E2E8F0"
COLOR_BORDER_SOFT = "#E9EDF3"
COLOR_TEXT = "#0F172A"
COLOR_TEXT_DIM = "#64748B"
COLOR_TEXT_FAINT = "#94A3B8"
COLOR_SLATE = "#334E68"
COLOR_ACCENT = "#2563EB"
COLOR_ACCENT_DIM = "#EFF6FF"
COLOR_ACCENT_DARK = "#1D4ED8"
COLOR_RED = "#E11D2A"
COLOR_RED_DARK = "#C11623"
COLOR_RED_DIM = "#FEE4E2"
COLOR_GREEN = "#16803C"
COLOR_GREEN_DARK = "#116230"
COLOR_GREEN_DIM = "#E6F4EA"
COLOR_AMBER = "#D97706"
COLOR_ORANGE = "#C2410C"
COLOR_PLACEHOLDER = COLOR_CARD_2
RADIUS = 16
RADIUS_SM = 10

# Shared toolbar layout constants — one gap size used between every toolbar
# section (Playback / Detection / Configuration / Calibrate) so the bar
# reads as a single evenly-spaced system instead of ad hoc gaps per row.
TOOLBAR_GAP = 24
TOOLBAR_GAP_HALF = 12
DIVIDER_LEN = 24

FONT_FAMILY = "Segoe UI"
FONT_STACK = f'"Inter", "{FONT_FAMILY}", "Helvetica Neue", Arial, sans-serif'

STYLE_SHEET = f"""
QMainWindow {{ background-color: {COLOR_BG}; }}
QWidget {{ color: {COLOR_TEXT}; font-family: {FONT_STACK}; font-size: 15px; }}

/* Global QLabel reset — IMPORTANT: once a QLabel gets ANY stylesheet rule
   applied, Qt's CSS engine takes over its full box model. Without an
   explicit border/background here, Qt silently draws a default QFrame
   border around plain text labels (this caused the boxed-looking
   "Video preview" / "No alerts" text in earlier builds). Setting both
   explicitly to none/transparent as the baseline prevents that for every
   QLabel in the app, and per-widget styles below simply override color. */
QLabel {{ border: none; background: transparent; }}

QGroupBox {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.4px;
    color: {COLOR_SLATE};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {COLOR_SLATE}; }}

/* Default buttons — large, rounded, consistent height/padding, bold label. */
QPushButton {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 19px;
    padding: 0px 16px;
    min-height: 38px;
    color: {COLOR_SLATE};
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.2px;
}}

QPushButton:hover {{
    background-color: {COLOR_CARD_2};
    border-color: #C7D0DD;
}}

QPushButton:pressed {{
    background-color: #DCE2EA;
}}

QPushButton:disabled {{
    color: {COLOR_TEXT_FAINT};
    background-color: {COLOR_BG_RAISED};
    border-color: {COLOR_BORDER_SOFT};
}}


/* Standard Upload button */
QPushButton#uploadBtn {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    color: #FFFFFF;
    font-weight: 700;
}}

QPushButton#uploadBtn:hover {{
    background-color: {COLOR_ACCENT_DARK};
    border-color: {COLOR_ACCENT_DARK};
}}

QPushButton#uploadBtn:disabled {{
    background-color: #A9C0F5;
    border-color: #A9C0F5;
    color: #F0F4FE;
}}


/* =========================================================
   HEADER UPLOAD BUTTON — SOLID RED
   ========================================================= */

QPushButton#headerUploadBtn {{
    background-color: #E11D2A;
    border: 1px solid #E11D2A;
    color: #E11D2A;          /* ← text is now red */
    font-weight: 700;
    padding: 0px 26px;
    min-height: 44px;
    max-height: 44px;
    min-width: 176px;
    font-size: 14px;
    letter-spacing: 0.3px;
    border-radius: 10px;
}}

QPushButton#headerUploadBtn:hover {{
    background-color: {COLOR_RED_DARK};
    border: 1px solid {COLOR_RED_DARK};
    color: #FFFFFF;
}}

QPushButton#headerUploadBtn:pressed {{
    background-color: #A01220;
    border: 1px solid #A01220;
    color: #FFFFFF;
}}

/* Keep the button SOLID RED even when disabled */
QPushButton#headerUploadBtn:disabled {{
    background-color: #E11D2A;
    border: 1px solid #E11D2A;
    color: #FFFFFF;
}}


/* =========================================================
   TOOLBAR ACTION BUTTONS
   ========================================================= */

/* Stop / destructive action */
QPushButton#dangerBtn {{
    background-color: {COLOR_RED_DIM};
    border: 1px solid #FCC9C4;
    color: {COLOR_RED};
    font-weight: 700;
    border-radius: 19px;
    min-height: 38px;
    padding: 0px 18px;
}}

QPushButton#dangerBtn:hover {{
    background-color: #FDD9D5;
    border-color: {COLOR_RED};
}}

QPushButton#dangerBtn:pressed {{
    background-color: #F9C2BC;
}}

QPushButton#dangerBtn:disabled {{
    color: {COLOR_TEXT_FAINT};
    border-color: {COLOR_BORDER_SOFT};
    background-color: {COLOR_CARD_2};
}}


/* Neutral actions */
QPushButton#neutralBtn {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    color: {COLOR_SLATE};
    border-radius: 19px;
    min-height: 38px;
    padding: 0px 18px;
    font-weight: 700;
}}

QPushButton#neutralBtn:hover {{
    background-color: {COLOR_CARD_2};
    border-color: #C7D0DD;
}}

QPushButton#neutralBtn:pressed {{
    background-color: #DCE2EA;
}}

QPushButton#neutralBtn:disabled {{
    color: {COLOR_TEXT_FAINT};
    border-color: {COLOR_BORDER_SOFT};
    background-color: {COLOR_CARD_2};
}}


/* Accent actions */
QPushButton#accentBtn {{
    background-color: {COLOR_ACCENT_DIM};
    border: 1px solid #BFDBFE;
    color: {COLOR_ACCENT};
    font-weight: 700;
    border-radius: 19px;
    min-height: 38px;
    padding: 0px 18px;
}}

QPushButton#accentBtn:hover {{
    background-color: #DBEAFE;
    border-color: {COLOR_ACCENT};
}}

QPushButton#accentBtn:pressed {{
    background-color: #C7DDFC;
}}

QPushButton#accentBtn:disabled {{
    color: {COLOR_TEXT_FAINT};
    border-color: {COLOR_BORDER_SOFT};
    background-color: {COLOR_CARD_2};
}}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{
    background-color: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 10px 14px;
    min-height: 44px;
    color: {COLOR_TEXT};
    font-size: 14px;
    font-weight: 600;
    selection-background-color: {COLOR_ACCENT};
}}
QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border-color: #C7D0DD; background-color: {COLOR_CARD_2};
}}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
    border-color: {COLOR_ACCENT};
}}

/* Toolbar field VALUES (Severity/Device/Profile dropdowns + the Calibration
   px/m stepper) are visually a step lighter than form fields elsewhere
   (e.g. the Calibration dialog) — regular/medium weight rather than the
   bold 600 used app-wide, per the typography hierarchy: section label >
   field label > field value. */
QComboBox#toolbarField, QDoubleSpinBox#toolbarField {{
    font-weight: 500;
    color: {COLOR_TEXT};
}}

/* Drop-down button area + a clean chevron-style arrow (no image asset
   needed — Qt's box model supports the classic border-triangle trick on
   pseudo-elements). Replaces the ugly native OS arrow. Sized down and
   given a softer, muted-gray stroke color so it reads as a deliberate
   chevron affordance rather than a flat placeholder dash. */
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox::down-arrow {{
    width: 0px;
    height: 0px;
    border-left: 4.5px solid transparent;
    border-right: 4.5px solid transparent;
    border-top: 5.5px solid {COLOR_TEXT_DIM};
    margin-right: 13px;
}}
QComboBox::down-arrow:on {{ border-top-color: {COLOR_ACCENT}; }}
QComboBox::down-arrow:disabled {{ border-top-color: {COLOR_TEXT_FAINT}; }}

/* Polished dropdown popup menu — the border-radius here now clips the
   actual popup window (see style_combo_popup()), not just inner content. */
QComboBox QAbstractItemView {{
    background-color: {COLOR_CARD}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 6px;
    selection-background-color: {COLOR_ACCENT_DIM}; selection-color: {COLOR_ACCENT_DARK};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    min-height: 34px; padding: 4px 12px; border-radius: 8px; font-weight: 600;
}}

/* Clean spinbox up/down buttons for the Calibration field — visible
   pill-shaped stepper with crisp CSS-triangle up/down arrows (no image
   assets needed). */
QDoubleSpinBox, QSpinBox {{ padding-right: 30px; }}

QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 24px; height: 19px; margin: 3px 4px 0 0;
    border: 1px solid {COLOR_BORDER}; border-radius: 6px;
    background: {COLOR_CARD_2};
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover {{
    background: {COLOR_ACCENT_DIM}; border-color: {COLOR_ACCENT};
}}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed {{ background: #DBEAFE; }}

QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 24px; height: 19px; margin: 0 4px 3px 0;
    border: 1px solid {COLOR_BORDER}; border-radius: 6px;
    background: {COLOR_CARD_2};
}}
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background: {COLOR_ACCENT_DIM}; border-color: {COLOR_ACCENT};
}}
QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{ background: #DBEAFE; }}

QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid {COLOR_TEXT_DIM};
}}
QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{ border-bottom-color: {COLOR_ACCENT_DARK}; }}

QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {COLOR_TEXT_DIM};
}}
QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{ border-top-color: {COLOR_ACCENT_DARK}; }}
QCheckBox {{ color: {COLOR_TEXT}; spacing: 10px; font-weight: 600; font-size: 14px; border: none; background: transparent; }}
QCheckBox::indicator {{
    width: 20px; height: 20px; border-radius: 5px;
    border: 1.5px solid #CBD5E1; background: {COLOR_CARD};
}}
QCheckBox::indicator:checked {{ background-color: {COLOR_ACCENT}; border-color: {COLOR_ACCENT}; }}
QCheckBox::indicator:hover {{ border-color: {COLOR_ACCENT}; }}

/* NOTE: the "Accident detection" toggle intentionally uses the SAME brand
   blue checked-state as every other checkbox (see QCheckBox::indicator
   above) — red is reserved for genuine alert/error states, not a passive
   "this feature is armed" toggle, so there is no red-specific override
   here anymore. */

QProgressBar {{
    border: none; border-radius: 4px; text-align: center;
    background-color: {COLOR_CARD_2}; height: 8px; color: transparent;
}}
QProgressBar::chunk {{ background-color: {COLOR_ACCENT}; border-radius: 4px; }}

/* Video scrub slider — flat track + small round drag handle, matching the
   reference Live Feed control strip (replaces the old flat QProgressBar). */
QSlider#videoScrub {{ height: 20px; }}
QSlider#videoScrub::groove:horizontal {{
    height: 5px; border-radius: 2.5px; background-color: {COLOR_CARD_2};
    border: 1px solid {COLOR_BORDER_SOFT};
}}
QSlider#videoScrub::sub-page:horizontal {{
    height: 5px; border-radius: 2.5px; background-color: {COLOR_ACCENT};
}}
QSlider#videoScrub::handle:horizontal {{
    background-color: {COLOR_ACCENT};
    border: 2px solid #FFFFFF;
    width: 12px; height: 12px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider#videoScrub::handle:horizontal:hover {{
    background-color: {COLOR_ACCENT_DARK};
    width: 14px; height: 14px; margin: -6px 0; border-radius: 8px;
}}
QSlider#videoScrub:disabled::groove:horizontal {{ background-color: {COLOR_BORDER_SOFT}; border-color: {COLOR_BORDER_SOFT}; }}
QSlider#videoScrub:disabled::sub-page:horizontal {{ background-color: {COLOR_TEXT_FAINT}; }}
QSlider#videoScrub:disabled::handle:horizontal {{ background-color: {COLOR_BG_RAISED}; border-color: {COLOR_BORDER}; }}

/* Shared circular control button for the video play/fullscreen icons —
   gives both a consistent hit-target, size and hover treatment so the
   control strip reads as one designed component instead of two loose
   flat icons. */
QPushButton#videoCtrlBtn {{
    background-color: {COLOR_CARD_2};
    border: 1px solid {COLOR_BORDER};
    border-radius: 16px;
    min-width: 32px; max-width: 32px;
    min-height: 32px; max-height: 32px;
    padding: 0px;
}}
QPushButton#videoCtrlBtn:hover {{ background-color: {COLOR_ACCENT_DIM}; border-color: {COLOR_ACCENT}; }}
QPushButton#videoCtrlBtn:pressed {{ background-color: #DBEAFE; }}
QPushButton#videoCtrlBtn:disabled {{ background-color: {COLOR_BG_RAISED}; border-color: {COLOR_BORDER_SOFT}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {COLOR_BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #C7D0DD; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}

QMessageBox {{ background-color: {COLOR_CARD}; }}
QMessageBox QLabel {{ color: {COLOR_TEXT}; }}
QDialog {{ background-color: {COLOR_BG}; }}
"""

# Status-dot colors used by MainWindow._set_status (presentation-only helper).
# Also drives the header "Status ● <state>" pill dot and the video-card
# title dot — the ONLY places red is used for status is genuine "error"/
# alert states, never for an idle/armed/neutral condition.
STATUS_COLORS = {
    "idle": COLOR_TEXT_FAINT,   # gray  — waiting / stopped
    "busy": COLOR_AMBER,        # amber — analyzing / working
    "ok": COLOR_GREEN,          # green — finished cleanly / connected
    "error": COLOR_RED,         # red   — error state
}

# Severity accent colors for Incident Report cards.
SEVERITY_ACCENTS = {
    "low": COLOR_GREEN,
    "medium": COLOR_AMBER,
    "high": COLOR_ORANGE,
    "critical": COLOR_RED,
}

# Incident-count badge treatment (header of the Incident Report panel).
# Kept as shared constants so the idle/reset state (_build_report_card,
# start_analysis) and the active state (on_alert) never drift out of sync.
BADGE_STYLE_IDLE = (
    f"background-color: {COLOR_CARD_2}; color: {COLOR_TEXT_DIM}; "
    f"border: 1px solid {COLOR_BORDER}; border-radius: 11px; "
    f"font-weight: 700; font-size: 11px; padding: 0 7px;"
)
BADGE_STYLE_ACTIVE = (
    f"background-color: {COLOR_RED}; color: #FFFFFF; "
    f"border: 1px solid {COLOR_RED}; border-radius: 11px; "
    f"font-weight: 700; font-size: 11px; padding: 0 7px;"
)


def bgr_to_qpixmap(frame, target_w=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_w:
        pix = pix.scaledToWidth(target_w, Qt.SmoothTransformation)
    return pix


def bgr_to_qpixmap_fit(frame, target_size):
    """Scale a BGR frame to fit entirely within target_size, preserving
    aspect ratio. The full frame remains visible; unused space is letterboxed
    with the display widget's background."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    if target_size.width() > 0 and target_size.height() > 0:
        pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix


def make_card_shadow(blur=28, y_offset=6, alpha=26):
    """A soft, low-key drop shadow used on top-level cards (video panel,
    report panel) for a bit of elevation — purely cosmetic, no effect on
    layout or any pipeline/worker behavior."""
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(0, y_offset)
    eff.setColor(QColor(23, 32, 51, alpha))
    return eff


def make_divider(vertical=True, length=22, thickness=1, color=COLOR_BORDER):
    """A small, explicitly-sized flat divider bar.

    NOTE: intentionally NOT a QFrame.VLine/HLine. Once a stylesheet is
    applied to a QFrame set to VLine/HLine, Qt's CSS engine can stop
    respecting the frame's line-thickness sizing and stretch it to fill
    the layout — which is what produced the stray horizontal bar artifact
    near the header status pill. A plain NoFrame QFrame with a hard-coded
    fixed size sidesteps that entirely.
    """
    line = QFrame()
    line.setFrameShape(QFrame.NoFrame)
    if vertical:
        line.setFixedSize(thickness, length)
    else:
        line.setFixedSize(length, thickness)
    line.setStyleSheet(f"background-color: {color}; border: none;")
    return line

def style_combo_popup(combo):
    """Makes a QComboBox's dropdown popup ACTUALLY have rounded corners.

    By default QSS border-radius on QAbstractItemView only rounds the
    content inside the popup — the popup's top-level window is still a
    plain opaque rectangle underneath, so you see a rounded rect drawn
    inside a square window (the "square outside, rounded inside" look).
    Making the popup window frameless + translucent lets the radius clip
    the actual window edges, which is what gives a clean, professional
    rounded dropdown like the reference design.
    """
    view = combo.view()
    popup = view.window()
    popup.setWindowFlags(popup.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
    popup.setAttribute(Qt.WA_TranslucentBackground, True)
    view.setAttribute(Qt.WA_TranslucentBackground, True)
    combo.setStyleSheet(combo.styleSheet())  # nudge Qt to reapply the QSS to the new popup window
# ---------------------------------------------------------------------------
# Vector icons — every icon in the app (upload, pause, stop, control room,
# camera, bell, info, play) is drawn with QPainter primitives (lines, arcs,
# paths) rather than emoji characters or raster images. This keeps the app
# a single self-contained file with no icon assets to ship, while still
# giving crisp, theme-colored, real vector iconography.
# ---------------------------------------------------------------------------

def render_icon_pixmap(draw_fn, size=20, color=COLOR_TEXT):
    """Render an icon draw function into a transparent QPixmap."""
    dpr = 2  # render at 2x for crisp icons on hi-DPI displays
    pix = QPixmap(size * dpr, size * dpr)
    pix.setDevicePixelRatio(dpr)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    draw_fn(painter, size, QColor(color))
    painter.end()
    return pix


def icon_from_draw(draw_fn, size=20, color=COLOR_TEXT):
    return QIcon(render_icon_pixmap(draw_fn, size, color))


def _draw_upload_arrow(p, size, color):
    pen = QPen(color, max(1.6, size * 0.10), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx = size * 0.5
    p.drawLine(QPointF(cx, size * 0.12), QPointF(cx, size * 0.58))
    path = QPainterPath()
    path.moveTo(cx - size * 0.22, size * 0.34)
    path.lineTo(cx, size * 0.10)
    path.lineTo(cx + size * 0.22, size * 0.34)
    p.drawPath(path)
    p.drawLine(QPointF(size * 0.16, size * 0.80), QPointF(size * 0.16, size * 0.90))
    p.drawLine(QPointF(size * 0.16, size * 0.90), QPointF(size * 0.84, size * 0.90))
    p.drawLine(QPointF(size * 0.84, size * 0.90), QPointF(size * 0.84, size * 0.80))


def _draw_pause(p, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    bar_w = size * 0.20
    gap = size * 0.16
    top = size * 0.18
    h = size * 0.64
    x1 = size * 0.5 - gap / 2 - bar_w
    x2 = size * 0.5 + gap / 2
    p.drawRoundedRect(QRectF(x1, top, bar_w, h), bar_w * 0.3, bar_w * 0.3)
    p.drawRoundedRect(QRectF(x2, top, bar_w, h), bar_w * 0.3, bar_w * 0.3)


def _draw_play(p, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    path = QPainterPath()
    path.moveTo(size * 0.28, size * 0.16)
    path.lineTo(size * 0.28, size * 0.84)
    path.lineTo(size * 0.82, size * 0.5)
    path.closeSubpath()
    p.drawPath(path)


def _draw_stop(p, size, color):
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    m = size * 0.20
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), size * 0.08, size * 0.08)


def _draw_sliders(p, size, color):
    pen = QPen(color, max(1.4, size * 0.09), Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    xs = [size * 0.26, size * 0.5, size * 0.74]
    knobs = [0.64, 0.36, 0.56]
    for x in xs:
        p.drawLine(QPointF(x, size * 0.10), QPointF(x, size * 0.90))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    for x, ky in zip(xs, knobs):
        p.drawEllipse(QPointF(x, size * ky), size * 0.10, size * 0.10)


def _draw_gear(p, size, color):
    """Small settings/configuration gear, used as a section-group icon for
    the Configuration cluster in the toolbar."""
    pen = QPen(color, max(1.3, size * 0.09), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    cx, cy, r = size * 0.5, size * 0.5, size * 0.22
    p.drawEllipse(QPointF(cx, cy), r, r)
    import math
    for i in range(8):
        ang = i * (math.pi / 4)
        x1 = cx + math.cos(ang) * (r + size * 0.06)
        y1 = cy + math.sin(ang) * (r + size * 0.06)
        x2 = cx + math.cos(ang) * (r + size * 0.18)
        y2 = cy + math.sin(ang) * (r + size * 0.18)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _draw_layers(p, size, color):
    """Playback/session-group icon (stacked layers) used above the
    Playback control cluster."""
    pen = QPen(color, max(1.3, size * 0.08), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    for dy in (0.30, 0.50, 0.70):
        path = QPainterPath()
        path.moveTo(size * 0.5, dy * size - size * 0.14)
        path.lineTo(size * 0.86, dy * size)
        path.lineTo(size * 0.5, dy * size + size * 0.14)
        path.lineTo(size * 0.14, dy * size)
        path.closeSubpath()
        if dy == 0.30:
            p.drawPath(path)


def _draw_shield(p, size, color):
    """Detection-group icon (shield) used above the Detection checkboxes."""
    pen = QPen(color, max(1.4, size * 0.09), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(size * 0.5, size * 0.08)
    path.lineTo(size * 0.86, size * 0.22)
    path.lineTo(size * 0.86, size * 0.52)
    path.cubicTo(size * 0.86, size * 0.76, size * 0.68, size * 0.90, size * 0.5, size * 0.96)
    path.cubicTo(size * 0.32, size * 0.90, size * 0.14, size * 0.76, size * 0.14, size * 0.52)
    path.lineTo(size * 0.14, size * 0.22)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(size * 0.36, size * 0.5), QPointF(size * 0.46, size * 0.62))
    p.drawLine(QPointF(size * 0.46, size * 0.62), QPointF(size * 0.66, size * 0.38))


def _draw_camera_off(p, size, color):
    # Consistent, polished stroke
    stroke = max(1.8, size * 0.065)
    pen = QPen(
        color,
        stroke,
        Qt.SolidLine,
        Qt.RoundCap,
        Qt.RoundJoin
    )

    p.setPen(pen)
    p.setBrush(Qt.NoBrush)

    # ── Camera body ─────────────────────────────────────
    body_x = size * 0.16
    body_y = size * 0.30
    body_w = size * 0.50
    body_h = size * 0.40
    radius = size * 0.07

    body = QRectF(
        body_x,
        body_y,
        body_w,
        body_h
    )

    p.drawRoundedRect(
        body,
        radius,
        radius
    )

    # ── Camera lens / right-side triangle ───────────────
    lens = QPainterPath()

    lens.moveTo(
        body.right(),
        size * 0.40
    )

    lens.lineTo(
        size * 0.86,
        size * 0.27
    )

    lens.lineTo(
        size * 0.86,
        size * 0.73
    )

    lens.lineTo(
        body.right(),
        size * 0.60
    )

    lens.closeSubpath()

    p.drawPath(lens)

    # ── Diagonal "off" slash ────────────────────────────
    p.drawLine(
        QPointF(size * 0.14, size * 0.12),
        QPointF(size * 0.86, size * 0.88)
    )


def _draw_fullscreen(p, size, color):
    """Compact expand/fullscreen corners icon for the video control strip."""
    pen = QPen(color, max(1.4, size * 0.11), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    m = size * 0.14   # margin from edge
    a = size * 0.30   # arm length

    # top-left corner
    p.drawLine(QPointF(m, m + a), QPointF(m, m))
    p.drawLine(QPointF(m, m), QPointF(m + a, m))
    # top-right corner
    p.drawLine(QPointF(size - m - a, m), QPointF(size - m, m))
    p.drawLine(QPointF(size - m, m), QPointF(size - m, m + a))
    # bottom-left corner
    p.drawLine(QPointF(m, size - m - a), QPointF(m, size - m))
    p.drawLine(QPointF(m, size - m), QPointF(m + a, size - m))
    # bottom-right corner
    p.drawLine(QPointF(size - m - a, size - m), QPointF(size - m, size - m))
    p.drawLine(QPointF(size - m, size - m), QPointF(size - m, size - m - a))


def _draw_bell(p, size, color):
    pen = QPen(color, max(1.6, size * 0.05), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(size * 0.30, size * 0.64)
    path.cubicTo(size * 0.30, size * 0.28, size * 0.70, size * 0.28, size * 0.70, size * 0.64)
    path.lineTo(size * 0.80, size * 0.76)
    path.lineTo(size * 0.20, size * 0.76)
    path.closeSubpath()
    p.drawPath(path)
    p.drawArc(QRectF(size * 0.40, size * 0.74, size * 0.20, size * 0.16), 0, -180 * 16)
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QPointF(size * 0.5, size * 0.18), size * 0.035, size * 0.035)


def _draw_info(p, size, color):
    pen = QPen(color, max(1.4, size * 0.10), Qt.SolidLine, Qt.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(size * 0.08, size * 0.08, size * 0.84, size * 0.84))
    p.drawLine(QPointF(size * 0.5, size * 0.46), QPointF(size * 0.5, size * 0.72))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    p.drawEllipse(QPointF(size * 0.5, size * 0.30), size * 0.05, size * 0.05)


def _draw_incident_outline(p, size, color):
    """Compact outlined incident/alert glyph (circle + exclamation), sized
    for a small light container. Replaces the oversized filled bell used
    previously in the Incident Report empty state — deliberately distinct
    from the header notification bell (_draw_bell)."""
    pen = QPen(color, max(1.3, size * 0.10), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(size * 0.09, size * 0.09, size * 0.82, size * 0.82))
    p.drawLine(QPointF(size * 0.5, size * 0.33), QPointF(size * 0.5, size * 0.58))
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(color))
    p.drawEllipse(QPointF(size * 0.5, size * 0.69), size * 0.045, size * 0.045)


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
                 clip_dir=CLIP_DIR, camera_cfg=None, run_accident=True,
                 run_violence=False, parent=None):
        super().__init__(parent)
        self.source_path = source_path
        self.device = device
        self.px_per_meter = px_per_meter
        self.alert_display_seconds = alert_display_seconds
        self.min_severity = min_severity
        self.camera_id = camera_id
        self.location = location
        self.clip_dir = clip_dir
        self.camera_cfg = camera_cfg
        self.run_accident = run_accident
        self.run_violence = run_violence
        self.fps = 25.0  # updated with the real source fps once run() opens the video
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
            self.fps = fps  # expose to the UI thread for elapsed/total time labels
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            frame_interval = 1.0 / fps

            pipeline = None
            if self.run_accident:
                camera_cfg = self.camera_cfg or CameraConfig(
                    camera_id=self.camera_id, location_name=self.location)
                # Manual px/m (the "Calibration" field) feeds the pixel-fallback
                # scale: the overlay's SpeedEstimator consumes m/s from history,
                # so the manual scale lands here as meter_per_pixel (ignored by
                # the ML estimator when a homography/profile is present).
                if self.px_per_meter:
                    camera_cfg.meter_per_pixel = 1.0 / self.px_per_meter
                pipeline = AccidentPipeline(
                    detector=Detector(device=self.device),
                    heuristic_cfg=HeuristicConfig(),
                    camera_cfg=camera_cfg,
                    dispatch_cfg=DispatchConfig(dashboard_log_path=ALERTS_LOG),
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
                    dispatch_cfg=DispatchConfig(dashboard_log_path=ALERTS_LOG),
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
                                 active_alerts, speed_estimator,
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
    """A small clickable screenshot thumbnail — styled like a recessed
    surface (bg-panel-2, soft border, small radius)."""

    def __init__(self, path=None, size=(94, 68)):
        super().__init__()
        self._path = None
        self.setFixedSize(*size)
        self.setScaledContents(True)
        self.setStyleSheet(
            f"background-color: {COLOR_CARD_2}; border: 1px solid {COLOR_BORDER_SOFT}; "
            f"border-radius: {RADIUS_SM}px; color: {COLOR_TEXT_FAINT}; font-size: 11px;"
        )
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

    def _open(self, _event):
        if not (self._path and os.path.exists(self._path)):
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Screenshot")
        dlg.setStyleSheet(f"background-color: {COLOR_BG};")
        pix = QPixmap(self._path)
        if pix.width() > 900:
            pix = pix.scaledToWidth(900, Qt.SmoothTransformation)
        lbl = QLabel()
        lbl.setPixmap(pix)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addWidget(lbl)
        dlg.exec()


class AlertCard(QFrame):
    """One entry in the side-panel report. Styled with a bg-panel-2 surface,
    soft border, 3px severity-colored left edge, plus the pre/impact/post
    filmstrip and a clip button."""

    def __init__(self, payload, shots, parent=None):
        super().__init__(parent)
        self.alert_id = payload.alert_id
        accent = SEVERITY_ACCENTS.get(payload.severity, COLOR_SLATE)
        self.setStyleSheet(
            f"AlertCard {{ background-color: {COLOR_CARD_2}; "
            f"border: 1px solid {COLOR_BORDER_SOFT}; border-left: 3px solid {accent}; "
            f"border-radius: {RADIUS_SM}px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        title_row = QHBoxLayout()
        kind = QLabel(payload.kind.replace('_', ' ').title())
        kind.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: 700; font-size: 13px; border: none; background: transparent;")
        title_row.addWidget(kind)
        title_row.addStretch(1)
        sev = QLabel(payload.severity.upper())
        sev.setStyleSheet(f"color: {accent}; font-weight: 700; font-size: 11.5px; letter-spacing: 0.3px; border: none; background: transparent;")
        title_row.addWidget(sev)
        outer.addLayout(title_row)

        ts = datetime.now().strftime("%H:%M:%S")
        subtitle = QLabel(
            f"t={payload.timestamp:.1f}s  ·  tracks {payload.track_ids}  ·  {ts}  ·  "
            f"→ {', '.join(c.replace('_', ' ') for c in payload.channels)}"
        )
        subtitle.setStyleSheet(f"color: {COLOR_TEXT_FAINT}; font-size: 11px; border: none; background: transparent;")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        strip = QHBoxLayout()
        strip.setSpacing(8)
        self.thumb_before = Thumb(shots.get("before"))
        self.thumb_impact = Thumb(shots.get("impact"))
        self.thumb_after = Thumb(shots.get("after"))
        for lbl, thumb in (("before", self.thumb_before), ("impact", self.thumb_impact),
                           ("after", self.thumb_after)):
            col = QVBoxLayout()
            col.setSpacing(3)
            col.addWidget(thumb)
            cap = QLabel(lbl)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet(f"color: {COLOR_TEXT_FAINT}; font-size: 10px; font-weight: 600; border: none; background: transparent;")
            col.addWidget(cap)
            strip.addLayout(col)
        strip.addStretch(1)
        outer.addLayout(strip)

        self.clip_btn = QPushButton("Play clip")
        self.clip_btn.setObjectName("neutralBtn")
        self.clip_btn.setIcon(icon_from_draw(_draw_play, size=11, color=COLOR_SLATE))
        self.clip_btn.setIconSize(QSize(11, 11))
        self.clip_btn.setToolTip("Open the saved pre/post-impact video clip for this alert")
        self.clip_btn.setEnabled(False)
        self.clip_btn.setStyleSheet(
            f"QPushButton#neutralBtn {{ min-height: 32px; padding: 0 14px; font-size: 11.5px; "
            f"background-color: {COLOR_CARD}; }}"
        )
        self.clip_btn.clicked.connect(lambda: self._play_clip())
        outer.addWidget(self.clip_btn)

    def set_after_shot(self, path):
        self.thumb_after.set_path(path)

    def set_clip_path(self, path):
        self._clip_path = path
        ok = bool(path and os.path.exists(path))
        self.clip_btn.setEnabled(ok)
        self.clip_btn.setText("Play clip" if ok else "Clip not found")

    def _play_clip(self):
        cpath = getattr(self, "_clip_path", None)
        if not cpath or not os.path.exists(cpath):
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(cpath))


class CalibrationDialog(QDialog):
    """
    GUI homography calibration flow:

        1. Load a frame (from a video file or image) — any frame where you
           can see 4+ ground-plane reference points (lane edges, crosswalk
           corners, a measured rectangle).
        2. Click each reference point on the image; enter its real-world
           (x, y) in meters and press "Add point". The world frame is any
           consistent origin/axes on the ground plane (e.g. origin at one
           corner of a marked lane, x = across the road, y = along it).
        3. "Compute & preview" builds the homography (reuses the math from
           tools/calibrate_camera.py) and shows the bird's-eye view — check
           that known-straight lines look straight there.
        4. "Save profile" writes a camera profile JSON (homography points,
           camera id/location, calibration note) that demo.py / the GUI
           pipeline can load directly.

    A profile's homography is only as good as the reference points: pick
    points on the ground plane (road surface), spread across the region of
    the frame where you need accurate speeds, with distances you measured
    or know (e.g. standard Indian lane width 3.5 m).
    """

    profile_saved = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.setStyleSheet(f"QDialog {{ background-color: {COLOR_BG}; }}")
        self.resize(1150, 720)
        self._frame = None
        self._points = []     # list of {"px":, "py":, "wx":, "wy":}
        self._pending = None  # (px, py) of the last image click
        self._H = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(10)
        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        src_row = QHBoxLayout()
        self.src_btn = QPushButton("Load Frame (video/image)")
        self.src_btn.setObjectName("neutralBtn")
        self.src_btn.clicked.connect(self.on_load_frame)
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 100000)
        self.frame_spin.setValue(0)
        src_row.addWidget(self.src_btn)
        src_row.addWidget(QLabel("Frame #"))
        src_row.addWidget(self.frame_spin)
        src_row.addStretch(1)
        left.addLayout(src_row)

        self.frame_view = ClickableImage("Click 4+ ground-plane points on the frame")
        self.frame_view.clicked.connect(self.on_image_click)
        left.addWidget(self.frame_view, 1)

        self.point_label = QLabel("Points: 0 / 4+ required")
        self.point_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-weight: 600;")
        left.addWidget(self.point_label)

        hint = QLabel("World coords are in meters; pick any consistent origin/axes "
                      "(e.g. x = across road, y = along road).")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLOR_TEXT_FAINT}; font-size: 11px;")
        right.addWidget(hint)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("World X (m)"))
        self.wx_spin = QDoubleSpinBox()
        self.wx_spin.setRange(-10000.0, 10000.0)
        self.wx_spin.setDecimals(2)
        coord_row.addWidget(self.wx_spin)
        coord_row.addWidget(QLabel("World Y (m)"))
        self.wy_spin = QDoubleSpinBox()
        self.wy_spin.setRange(-10000.0, 10000.0)
        self.wy_spin.setDecimals(2)
        coord_row.addWidget(self.wy_spin)
        self.add_btn = QPushButton("Add point")
        self.add_btn.setObjectName("uploadBtn")
        self.add_btn.clicked.connect(self.on_add_point)
        coord_row.addWidget(self.add_btn)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setObjectName("neutralBtn")
        self.undo_btn.clicked.connect(self.on_undo)
        coord_row.addWidget(self.undo_btn)
        right.addLayout(coord_row)

        self.compute_btn = QPushButton("Compute homography + preview bird's-eye")
        self.compute_btn.setObjectName("neutralBtn")
        self.compute_btn.clicked.connect(self.on_compute)
        right.addWidget(self.compute_btn)

        self.preview_view = ClickableImage("Bird's-eye preview appears here")
        self.preview_view.setMinimumSize(400, 260)
        right.addWidget(self.preview_view, 1)

        meta_box = QGroupBox("PROFILE METADATA")
        meta = QVBoxLayout(meta_box)
        meta.setSpacing(8)
        meta.addWidget(QLabel("Camera ID"))
        self.camera_id_edit = QLineEdit("CAM-01")
        meta.addWidget(self.camera_id_edit)
        meta.addWidget(QLabel("Location"))
        self.location_edit = QLineEdit("Unnamed camera")
        meta.addWidget(self.location_edit)
        mpp_row = QHBoxLayout()
        mpp_row.addWidget(QLabel("Fallback px/m (0 = keep config default)"))
        self.mpp_spin = QDoubleSpinBox()
        self.mpp_spin.setRange(0.0, 500.0)
        self.mpp_spin.setDecimals(2)
        self.mpp_spin.setValue(0.0)
        mpp_row.addWidget(self.mpp_spin)
        meta.addLayout(mpp_row)
        meta.addWidget(QLabel("Calibration note (what did you measure/assume?)"))
        self.note_edit = QLineEdit("")
        self.note_edit.setPlaceholderText("e.g. lane width 3.5 m (standard Indian lane) assumed")
        meta.addWidget(self.note_edit)
        right.addWidget(meta_box)

        save_row = QHBoxLayout()
        save_row.setSpacing(8)
        self.save_btn = QPushButton("Save profile JSON")
        self.save_btn.setObjectName("uploadBtn")
        self.save_btn.clicked.connect(self.on_save)
        self.use_btn = QPushButton("Use for this session")
        self.use_btn.setObjectName("neutralBtn")
        self.use_btn.setEnabled(False)
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.use_btn)
        right.addLayout(save_row)

    def on_load_frame(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video or image", "",
            "Video/Image files (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All files (*)"
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png"):
            frame = cv2.imread(path)
        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                QMessageBox.critical(self, "Error", f"Could not open: {path}")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_spin.value())
            ok, frame = cap.read()
            cap.release()
            if not ok:
                QMessageBox.critical(self, "Error", f"Could not read frame {self.frame_spin.value()} from {path}")
                return
        self._frame = frame
        self._points = []
        self._pending = None
        self._H = None
        self.use_btn.setEnabled(False)
        self._refresh_points()
        self.frame_view.set_frame(frame)
        self.preview_view.set_frame(np.zeros((180, 320, 3), dtype=np.uint8))

    def on_image_click(self, px, py):
        self._pending = (px, py)
        self.frame_view.set_frame(self._draw_points())

    def on_add_point(self):
        if self._frame is None:
            return
        if self._pending is None:
            QMessageBox.information(self, "No point clicked",
                                    "Click a point on the frame first, then set its world coords.")
            return
        self._points.append({
            "px": self._pending[0], "py": self._pending[1],
            "wx": self.wx_spin.value(), "wy": self.wy_spin.value(),
        })
        self._pending = None
        self._H = None
        self._refresh_points()
        self.frame_view.set_frame(self._draw_points())

    def on_undo(self):
        if self._points:
            self._points.pop()
            self._H = None
            self._refresh_points()
            self.frame_view.set_frame(self._draw_points())
        else:
            self._pending = None
            self.frame_view.set_frame(self._draw_points())

    def on_compute(self):
        if self._frame is None:
            return
        if len(self._points) < 4:
            QMessageBox.warning(self, "Not enough points",
                                "Need at least 4 ground-plane points (more is better).")
            return
        try:
            raw = [(p["px"], p["py"], p["wx"], p["wy"]) for p in self._points]
            H, src, dst = build_homography(raw)
        except SystemExit as e:
            QMessageBox.critical(self, "Homography failed", str(e))
            return
        self._H = H
        import tempfile
        preview_path = os.path.join(tempfile.gettempdir(), "vista_calibration_preview.png")
        render_birdseye_preview(self._frame, H, preview_path, dst)
        preview = cv2.imread(preview_path)
        if preview is not None:
            self.preview_view.set_frame(preview)
        self.use_btn.setEnabled(True)

    def on_save(self):
        if self._frame is None:
            return
        if self._H is None:
            self.on_compute()
            if self._H is None:
                return
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_profiles")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save camera profile", os.path.join(default_dir, "CAM-01.json"),
            "JSON (*.json)"
        )
        if not path:
            return
        cfg = CameraConfig(
            camera_id=self.camera_id_edit.text().strip() or "CAM-01",
            location_name=self.location_edit.text().strip() or "Unnamed camera",
            homography_src_points=[[p["px"], p["py"]] for p in self._points],
            homography_dst_points=[[p["wx"], p["wy"]] for p in self._points],
        )
        if self.mpp_spin.value() > 0:
            cfg.meter_per_pixel = self.mpp_spin.value()
        note = self.note_edit.text().strip()
        save_profile(path, cfg, calibration_note=note)
        self.profile_saved.emit(path)
        QMessageBox.information(
            self, "Profile saved",
            f"Saved to:\n{path}\n\nLoad it with Camera Profile -> Use, or "
            f"python demo.py --camera-profile {path}"
        )

    def _draw_points(self):
        frame = self._frame.copy()
        for i, p in enumerate(self._points):
            cv2.circle(frame, (p["px"], p["py"]), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"#{i+1} ({p['wx']:.2f},{p['wy']:.2f})", (p["px"] + 8, p["py"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self._pending is not None:
            cv2.circle(frame, self._pending, 5, (0, 255, 255), -1)
        return frame

    def _refresh_points(self):
        self.point_label.setText(f"Points: {len(self._points)} / 4+ required")


class ClickableImage(QLabel):
    """A QLabel that reports clicks mapped back to original-frame pixel
    coordinates (compensates for the displayed pixmap's scaling)."""

    clicked = pyqtSignal(int, int)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._frame_w = None
        self._scale_x = 1.0
        self._scale_y = 1.0
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background-color: {COLOR_BG_RAISED}; color: {COLOR_TEXT_FAINT}; "
            f"border: 1px dashed {COLOR_BORDER}; border-radius: {RADIUS_SM}px;"
        )
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    def set_frame(self, frame, target_w=None):
        self._frame_w = frame.shape[1]
        self._frame_h = frame.shape[0]
        pix = bgr_to_qpixmap(frame, target_w=target_w or (self.width() or 640))
        self._scale_x = self._frame_w / max(1.0, pix.width())
        self._scale_y = self._frame_h / max(1.0, pix.height())
        self.setPixmap(pix)

    def mousePressEvent(self, event):
        if self._frame_w is None:
            return
        x = int(event.pos().x() * self._scale_x)
        y = int(event.pos().y() * self._scale_y)
        if 0 <= x < self._frame_w and 0 <= y < self._frame_h:
            self.clicked.emit(x, y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VISTA — Accident Detection")
        self.resize(1520, 900)
        self.worker = None
        self.alert_cards = {}
        self.alert_count = 0
        self.control_proc = None
        self.camera_cfg = None           # loaded camera profile (None = defaults)
        self.camera_profile_path = None
        self._last_frame = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())

        body = QWidget()
        body.setStyleSheet(f"background-color: {COLOR_BG};")
        body_v = QVBoxLayout(body)
        body_v.setContentsMargins(28, 20, 28, 20)
        body_v.setSpacing(16)

        content_row = QHBoxLayout()
        content_row.setSpacing(20)
        content_row.addWidget(self._build_video_card(), 3)
        content_row.addWidget(self._build_report_card(), 0)
        body_v.addLayout(content_row, 1)

        body_v.addWidget(self._build_status_footer())

        root.addWidget(body, 1)

        # Initial idle status (presentation-only; same text as before, now
        # routed through the status-dot helper).
        self._set_status("Idle — Waiting for a video to analyze.", "idle")

    # ------------------------------------------------------------
    # Presentation-only UI builders
    # ------------------------------------------------------------

    def _build_header(self):
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLOR_CARD}; border-bottom: 1px solid {COLOR_BORDER};"
        )
        header.setFixedHeight(84)
        h = QHBoxLayout(header)
        h.setContentsMargins(32, 0, 32, 0)
        h.setSpacing(16)

        # --- Brand cluster: "VISTA" + divider + "Accident Detection", all on
        # one line, vertically centered in the bar. ---
        brand_row = QHBoxLayout()
        brand_row.setSpacing(14)
        brand_row.setContentsMargins(0, 0, 0, 0)

        brand = QLabel("VISTA")
        brand.setStyleSheet(
            f"color: {COLOR_TEXT}; letter-spacing: 0.4px; font-weight: 800; font-size: 26px; "
            f"border: none; background: transparent;"
        )
        brand_row.addWidget(brand)

        brand_row.addWidget(make_divider(vertical=True, length=26, thickness=1))

        subtitle = QLabel("Accident Detection")
        subtitle.setStyleSheet(
            f"color: {COLOR_TEXT_DIM}; font-size: 14px; font-weight: 600; "
            f"letter-spacing: 0.2px; border: none; background: transparent;"
        )
        brand_row.addWidget(subtitle)

        h.addLayout(brand_row)
        h.addStretch(1)

        # --- Right cluster: "Status  ● Idle" pill, then the primary Upload
        # Video CTA. Uses the same header_status_label + _set_status()
        # plumbing as before, so live status updates keep working unchanged.
        # The pill is plain text+dot only — there is no separate empty
        # "box" here to fill with red; red on this pill only ever appears
        # via STATUS_COLORS["error"] when _set_status(..., "error") fires
        # for a genuine error/alert condition. ---
        self.header_status_label = QLabel()
        self.header_status_label.setTextFormat(Qt.RichText)
        h.addWidget(self.header_status_label)

        h.addSpacing(22)

        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.setObjectName("headerUploadBtn")
        self.upload_btn.setIcon(icon_from_draw(_draw_upload_arrow, size=18, color="#FFFFFF"))
        self.upload_btn.setIconSize(QSize(18, 18))
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.clicked.connect(self.on_upload)
        h.addWidget(self.upload_btn)

        return header

    def _build_toolbar(self):
        bar = QWidget()
        bar.setStyleSheet(
            f"background-color: {COLOR_CARD}; border-bottom: 1px solid {COLOR_BORDER};"
        )
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(32, 18, 32, 18)
        outer.setSpacing(16)

        # --- Row 1: PLAYBACK cluster, then DETECTION cluster --------------
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        row1.addLayout(self._group_header(_draw_layers, "PLAYBACK"))
        row1.addSpacing(TOOLBAR_GAP_HALF)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("accentBtn")
        self.pause_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_ACCENT))
        self.pause_btn.setIconSize(QSize(15, 15))
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setIcon(icon_from_draw(_draw_stop, size=13, color=COLOR_RED))
        self.stop_btn.setIconSize(QSize(13, 13))
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)

        row1.addWidget(self.pause_btn)
        row1.addWidget(self.stop_btn)

        row1.addSpacing(TOOLBAR_GAP_HALF)
        row1.addWidget(make_divider(vertical=True, length=DIVIDER_LEN, thickness=1))
        row1.addSpacing(TOOLBAR_GAP_HALF)

        # "Control room" is a secondary, polished, neutral-styled action —
        # it opens a separate browser console, so it's visually distinct
        # from the primary playback controls but still fully wired up.
        self.control_room_btn = QPushButton("Control room")
        self.control_room_btn.setObjectName("neutralBtn")
        self.control_room_btn.setIcon(icon_from_draw(_draw_sliders, size=15, color=COLOR_SLATE))
        self.control_room_btn.setIconSize(QSize(15, 15))
        self.control_room_btn.setToolTip(
            "Start the control-room console (siren + clip playback + routed "
            "recipients) in a browser — it watches alerts.jsonl and vista_clips/"
        )
        self.control_room_btn.clicked.connect(self.on_open_control_room)
        row1.addWidget(self.control_room_btn)

        row1.addSpacing(TOOLBAR_GAP_HALF)
        row1.addWidget(make_divider(vertical=True, length=DIVIDER_LEN, thickness=1))
        row1.addSpacing(TOOLBAR_GAP_HALF)

        # Push everything from here on (the DETECTION cluster) to the far
        # right edge of the row.
        row1.addStretch(1)

        row1.addLayout(self._group_header(_draw_shield, "DETECTION"))
        row1.addSpacing(TOOLBAR_GAP_HALF)

        # Accident-detection is a plain checkbox, styled identically to the
        # Violence-detection checkbox: brand blue when checked. Red is
        # deliberately NOT used here — this is an armed/passive toggle, not
        # an alert — so it no longer carries a red-accent objectName.
        self.accident_check = QCheckBox("Accident detection")
        self.accident_check.setChecked(True)
        self.accident_check.setToolTip(
            "Detect collisions/accidents with the yolo11m vehicle branch. "
            "Runs EVERY frame (the heavy model). Untick it and run the "
            "violence branch alone for a smooth, faster violence-only test."
        )
        row1.addWidget(self.accident_check)

        self.violence_check = QCheckBox("Violence detection")
        self.violence_check.setToolTip(
            "Also run the pose-based violence/road-rage branch (yolo11n-pose, "
            "auto-downloaded on first run): close persons + aggressive limb "
            "motion (or sustained box overlap for distant CCTV fights) route "
            "alerts to the police control room."
        )
        row1.addWidget(self.violence_check)

        outer.addLayout(row1)

        outer.addWidget(make_divider(vertical=False, length=10_000, thickness=1))

        # --- Row 2: CONFIGURATION cluster, then Calibrate — one continuous
        # run of fields at a consistent TOOLBAR_GAP, "Calibrate…" placed
        # right after the Configuration cluster (no more oversized stretch
        # gap pushing it off to the far edge). -----------------------------
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        row2.addLayout(self._group_header(_draw_gear, "CONFIGURATION"))
        row2.addSpacing(TOOLBAR_GAP)

        row2.addWidget(self._toolbar_label("Severity"))
        self.severity_combo = QComboBox()
        self.severity_combo.setObjectName("toolbarField")
        self.severity_combo.addItems(["low", "medium", "high", "critical"])
        self.severity_combo.setCurrentText("low")
        self.severity_combo.setToolTip(
            "Alerts below this severity are treated as noise: no report card, "
            "no screenshots."
        )
        self.severity_combo.setMinimumWidth(170)
        self.severity_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        style_combo_popup(self.severity_combo)
        row2.addWidget(self.severity_combo)

        row2.addSpacing(TOOLBAR_GAP)
        row2.addWidget(self._toolbar_label("Device"))
        self.device_combo = QComboBox()
        self.device_combo.setObjectName("toolbarField")
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setMinimumWidth(150)
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        style_combo_popup(self.device_combo)
        row2.addWidget(self.device_combo)

        row2.addSpacing(TOOLBAR_GAP)
        row2.addWidget(self._toolbar_label("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("toolbarField")
        self.profile_combo.addItem("(default)")
        self.profile_combo.setToolTip(
            "Camera calibration profiles in camera_profiles/ (homography + "
            "stop zones + camera metadata). Pick one to run analysis with "
            "real-world speeds."
        )
        self.profile_combo.setMinimumWidth(190)
        self.profile_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        style_combo_popup(self.profile_combo)
        self._refresh_profile_combo()
        row2.addWidget(self.profile_combo)

        # "Browse…" is kept alive (still fully wired to on_profile_browse)
        # but hidden from the toolbar — the reference design drops it in
        # favor of the Profile dropdown + "Calibrate…" flow. Set it visible
        # again if you want the old control back.
        self.profile_browse_btn = QPushButton("Browse…")
        self.profile_browse_btn.setObjectName("neutralBtn")
        self.profile_browse_btn.clicked.connect(self.on_profile_browse)
        self.profile_browse_btn.setVisible(False)

        row2.addSpacing(TOOLBAR_GAP)
        row2.addWidget(self._toolbar_label("Calibration"))
        self.calib_spin = QDoubleSpinBox()
        self.calib_spin.setObjectName("toolbarField")
        self.calib_spin.setPrefix("px/m: ")
        self.calib_spin.setRange(0.0, 500.0)
        self.calib_spin.setValue(0.0)
        self.calib_spin.setMinimumWidth(190)
        self.calib_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.calib_spin.setToolTip(
            "Optional: real camera calibration (pixels per meter). "
            "0 = auto-estimate speed from each object's own size."
        )
        row2.addWidget(self.calib_spin)

        row2.addSpacing(TOOLBAR_GAP)
        row2.addWidget(make_divider(vertical=True, length=DIVIDER_LEN, thickness=1))
        row2.addSpacing(TOOLBAR_GAP)

        # "Calibrate…" is a normal secondary action now: same objectName
        # (and therefore identical height/padding/radius/font-weight) as
        # "Control room", and navy/slate text instead of red — red is no
        # longer used for anything but genuine alert/error states.
        self.calibrate_btn = QPushButton("Calibrate…")
        self.calibrate_btn.setObjectName("neutralBtn")
        self.calibrate_btn.setIcon(icon_from_draw(_draw_gear, size=14, color=COLOR_SLATE))
        self.calibrate_btn.setIconSize(QSize(14, 14))
        self.calibrate_btn.setCursor(Qt.PointingHandCursor)
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        row2.addWidget(self.calibrate_btn)

        row2.addStretch(1)

        outer.addLayout(row2)

        return bar

    @staticmethod
    def _group_header(draw_fn, text):
        """A small caps icon+label pair used to mark the start of a control
        cluster (PLAYBACK / DETECTION / CONFIGURATION) in the toolbar.
        Deliberately smaller/muted-gray/bold — one step below field labels
        in the toolbar's type hierarchy (section label < field label <
        field value)."""
        row = QHBoxLayout()
        row.setSpacing(7)
        icon = QLabel()
        icon.setPixmap(render_icon_pixmap(draw_fn, size=13, color=COLOR_TEXT_DIM))
        row.addWidget(icon)
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLOR_TEXT_DIM}; font-size: 11px; font-weight: 800; "
            f"letter-spacing: 0.8px; border: none; background: transparent;"
        )
        row.addWidget(lbl)
        return row

    @staticmethod
    def _toolbar_label(text):
        """Field labels (Severity / Device / Profile / Calibration) — medium
        weight, dark text: one step above field VALUES (regular weight, see
        QComboBox#toolbarField / QDoubleSpinBox#toolbarField), one step
        below the bold muted-gray section labels from _group_header."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 13px; font-weight: 600; "
            f"border: none; background: transparent;"
        )
        return lbl   

    def _build_video_card(self):
        card = QFrame()
        card.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS}px; }}"
        )
        card.setGraphicsEffect(make_card_shadow())
        v = QVBoxLayout(card)
        v.setContentsMargins(24, 20, 24, 22)
        v.setSpacing(16)

        # --- Title row: small status dot + bold title -----
        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        dot = QLabel()
        dot.setFixedSize(9, 9)
        dot.setStyleSheet(
            f"background-color: {COLOR_TEXT_FAINT}; border-radius: 4px; border: none;"
        )
        self.video_dot = dot
        title_row.addWidget(dot, alignment=Qt.AlignVCenter)

        title = QLabel("Video Preview")
        title.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 16px; font-weight: 700; "
            f"border: none; background: transparent;"
        )
        title_row.addWidget(title, alignment=Qt.AlignVCenter)
        title_row.addStretch(1)
        v.addLayout(title_row)

        display_wrap = QWidget()
        display_wrap.setStyleSheet(
            f"background-color: {COLOR_CARD_2}; border-radius: {RADIUS_SM}px;"
        )
        stack = QStackedLayout(display_wrap)
        stack.setContentsMargins(0, 0, 0, 0)
        self._video_stack = stack
        self.video_display_wrap = display_wrap

        # Empty-state copy/icon: crossed-out camera above "No video loaded" /
        # "Upload a video to begin analysis", with a direct Upload action so
        # the empty state isn't a dead end.
        self.video_empty_widget = self._make_empty_state(
            _draw_camera_off, "No video loaded", "Upload a video to begin analysis",
            bg=None, icon_color=COLOR_SLATE, icon_size=56,
            action_text="Upload Video", action_slot=self.on_upload,
        )
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: transparent; border: none;")

        stack.addWidget(self.video_empty_widget)
        stack.addWidget(self.video_label)
        stack.setCurrentIndex(0)

        display_wrap.setMinimumSize(680, 460)
        display_wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(display_wrap, 1)

        # Scrub row: flat play icon (no button chrome) + elapsed time +
        # draggable-handle slider + total time + fullscreen, matching the
        # reference's video-player control strip exactly.
        # Playback control bar: play, elapsed time, scrub slider, total time,
        # fullscreen — one QHBoxLayout, all children vertically centered on
        # a shared centerline, with a fixed row height so nothing drifts.
        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(12)
        scrub_row.setContentsMargins(0, 0, 0, 0)
        scrub_row.setAlignment(Qt.AlignVCenter)

        self.play_btn = QPushButton()
        self.play_btn.setObjectName("videoCtrlBtn")
        self.play_btn.setIcon(icon_from_draw(_draw_play, size=15, color=COLOR_TEXT_FAINT))
        self.play_btn.setIconSize(QSize(15, 15))
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.clicked.connect(self.on_pause)
        self.play_btn.setEnabled(False)
        scrub_row.addWidget(self.play_btn, 0, Qt.AlignVCenter)

        self.time_elapsed_label = QLabel("0:00")
        self.time_elapsed_label.setFixedWidth(34)
        self.time_elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_elapsed_label.setStyleSheet(
            f"color: {COLOR_TEXT_FAINT}; font-size: 12.5px; font-weight: 600; "
            f"border: none; background: transparent;"
        )
        scrub_row.addWidget(self.time_elapsed_label, 0, Qt.AlignVCenter)

        self.progress_bar = QSlider(Qt.Horizontal)
        self.progress_bar.setObjectName("videoScrub")
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setEnabled(False)  # display-only; driven by on_progress()
        scrub_row.addWidget(self.progress_bar, 1, Qt.AlignVCenter)

        self.time_total_label = QLabel("0:00")
        self.time_total_label.setFixedWidth(34)
        self.time_total_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.time_total_label.setStyleSheet(
            f"color: {COLOR_TEXT_FAINT}; font-size: 12.5px; font-weight: 600; "
            f"border: none; background: transparent;"
        )
        scrub_row.addWidget(self.time_total_label, 0, Qt.AlignVCenter)

        self.fullscreen_btn = QPushButton()
        self.fullscreen_btn.setObjectName("videoCtrlBtn")
        self.fullscreen_btn.setIcon(icon_from_draw(_draw_fullscreen, size=15, color=COLOR_TEXT_FAINT))
        self.fullscreen_btn.setIconSize(QSize(15, 15))
        self.fullscreen_btn.setCursor(Qt.PointingHandCursor)
        self.fullscreen_btn.setEnabled(False)
        scrub_row.addWidget(self.fullscreen_btn, 0, Qt.AlignVCenter)

        v.addLayout(scrub_row)

        return card

    def _build_report_card(self):
        card = QFrame()
        card.setFixedWidth(ALERT_PANEL_WIDTH)
        card.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS}px; }}"
        )
        card.setGraphicsEffect(make_card_shadow())
        v = QVBoxLayout(card)
        v.setContentsMargins(22, 18, 22, 20)
        v.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header = QLabel("Incident Report")
        header.setStyleSheet(
            f"color: {COLOR_TEXT}; letter-spacing: 0.1px; font-size: 15.5px; "
            f"font-weight: 800; border: none; background: transparent;"
        )
        header_row.addWidget(header)
        header_row.addStretch(1)
        self.alert_badge = QLabel("0")
        self.alert_badge.setAlignment(Qt.AlignCenter)
        self.alert_badge.setFixedHeight(22)
        self.alert_badge.setMinimumWidth(22)
        self.alert_badge.setStyleSheet(BADGE_STYLE_IDLE)
        header_row.addWidget(self.alert_badge)
        v.addLayout(header_row)

        v.addWidget(make_divider(vertical=False, length=10_000, thickness=1))

        self.summary_label = QLabel("No alerts yet.")
        self.summary_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 13px; font-weight: 600; border: none; background: transparent;")
        self.summary_label.setVisible(False)  # empty state covers this until first alert
        v.addWidget(self.summary_label)

        self.report_scroll = QScrollArea()
        self.report_scroll.setWidgetResizable(True)
        self.report_container = QWidget()
        self.report_container.setStyleSheet("background-color: transparent;")
        self.report_layout = QVBoxLayout(self.report_container)
        self.report_layout.setSpacing(10)
        self.report_layout.setContentsMargins(0, 6, 0, 6)

        self.report_empty_state = self._make_empty_state(
            _draw_incident_outline, "No incidents detected",
            "Upload and analyze footage to see\ndetected incidents here.",
            bg=COLOR_CARD_2, icon_color=COLOR_TEXT_DIM, circle_size=46, icon_size=20,
            container_radius=14, container_border=COLOR_BORDER_SOFT,
            title_size=20, subtitle_size=14, spacing=12,
        )
        self.report_layout.addStretch(1)
        self.report_layout.addWidget(self.report_empty_state)
        self.report_layout.addStretch(1)

        self.report_scroll.setWidget(self.report_container)
        v.addWidget(self.report_scroll, 1)

        return card

    def _build_status_footer(self):
        footer = QFrame()
        footer.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_CARD}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: {RADIUS}px; }}"
        )
        footer.setFixedHeight(52)
        h = QHBoxLayout(footer)
        h.setContentsMargins(20, 8, 20, 8)
        h.setSpacing(10)

        icon = QLabel()
        icon.setFixedSize(22, 22)
        icon.setAlignment(Qt.AlignCenter)
        icon.setPixmap(render_icon_pixmap(_draw_info, size=18, color=COLOR_TEXT_FAINT))
        h.addWidget(icon)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_DIM}; font-size: 13.5px; font-weight: 600; border: none; background: transparent;")
        h.addWidget(self.status_label, 1)

        return footer

    @staticmethod
    def _make_empty_state(draw_fn, title_text, subtitle_text, bg=None,
                           icon_color=COLOR_TEXT_FAINT, circle_size=72, icon_size=30,
                           action_text=None, action_slot=None,
                           container_radius=None, container_border=None,
                           title_size=18, subtitle_size=13, spacing=14):
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        v = QVBoxLayout(wrap)
        v.setSpacing(spacing)
        v.setAlignment(Qt.AlignCenter)

        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignCenter)
        if bg:
            icon_label.setFixedSize(circle_size, circle_size)
            # container_radius lets callers use a subtle rounded-square
            # "chip" (e.g. radius 12) instead of a full circle — a lighter,
            # more restrained treatment for compact empty states.
            radius = container_radius if container_radius is not None else circle_size // 2
            border_css = f"border: 1px solid {container_border};" if container_border else "border: none;"
            icon_label.setStyleSheet(
                f"background-color: {bg}; border-radius: {radius}px; {border_css}"
            )
        else:
            icon_label.setFixedSize(icon_size + 8, icon_size + 8)
            icon_label.setStyleSheet("background: transparent; border: none;")
        icon_label.setPixmap(render_icon_pixmap(draw_fn, size=icon_size, color=icon_color))
        icon_row = QHBoxLayout()
        icon_row.addStretch(1)
        icon_row.addWidget(icon_label)
        icon_row.addStretch(1)
        v.addLayout(icon_row)

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {COLOR_TEXT}; font-weight: 700; font-size: {title_size}px; "
            f"border: none; background: transparent;"
        )
        v.addWidget(title)

        subtitle = QLabel(subtitle_text)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {COLOR_TEXT_FAINT}; font-size: {subtitle_size}px; border: none; background: transparent;")
        subtitle.setWordWrap(True)
        v.addWidget(subtitle)

        # Optional inline call-to-action (e.g. "Upload Video" from the video
        # empty-state) so the empty state is actionable, not a dead end.
        if action_text and action_slot:
            action_btn = QPushButton(action_text)
            action_btn.setObjectName("accentBtn")
            action_btn.setCursor(Qt.PointingHandCursor)
            action_btn.clicked.connect(action_slot)
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            btn_row.addWidget(action_btn)
            btn_row.addStretch(1)
            v.addLayout(btn_row)

        return wrap

    # ------------------------------------------------------------

    def _set_status(self, text, kind="idle"):
        """Presentation-only helper: colors the status dot next to the
        status text, in both the footer's detailed status line and the
        header's compact status pill. 'kind' does not affect any
        pipeline/worker behavior — it only picks a color from
        STATUS_COLORS ('idle' gray, 'busy' amber, 'ok' green, 'error' red)."""
        dot = STATUS_COLORS.get(kind, STATUS_COLORS["idle"])
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setText(
            f"<span style='color:{dot};'>&#9679;</span>&nbsp;&nbsp;{text}"
        )

        short = {"idle": "Idle", "busy": "Analyzing", "ok": "Done", "error": "Error"}.get(kind, "Idle")
        # Status pill: small vertically-centered colored dot immediately
        # before the state word (gray=idle, amber=busy, green=ok, red=error
        # — i.e. red ONLY for a genuine error/alert state).
        self.header_status_label.setText(
            f"<span style='color:{COLOR_TEXT_FAINT}; font-size:12.5px; font-weight:600;'>Status</span>"
            f"&nbsp;&nbsp;<span style='color:{dot}; font-size:13px; vertical-align:middle;'>&#9679;</span>"
            f"&nbsp;<span style='color:{COLOR_TEXT}; font-weight:700; font-size:13px;'>{short}</span>"
        )
        if hasattr(self, "video_dot"):
            self.video_dot.setStyleSheet(
                f"background-color:{dot}; border-radius: 4px; border: none;"
            )

    def _set_playback_controls_enabled(self, enabled):
        """Keep the flat play/fullscreen icon buttons and their labels
        visually in sync with their enabled state. setEnabled() alone
        doesn't gray out a flat/borderless QPushButton's icon, so without
        this the scrub row looked identically "active" whether or not a
        video was actually loaded/playing."""
        icon_color = COLOR_SLATE if enabled else COLOR_TEXT_FAINT
        label_color = COLOR_SLATE if enabled else COLOR_TEXT_FAINT

        self.play_btn.setEnabled(enabled)
        self.fullscreen_btn.setEnabled(enabled)
        self.progress_bar.setEnabled(enabled)

        # Preserve whichever play/pause glyph is currently correct — callers
        # that need to flip the glyph (e.g. on_pause toggling play<->pause)
        # still set the icon explicitly right after calling this.
        
        current_draw = _draw_pause if getattr(self, "_is_playing", False) else _draw_play
        self.play_btn.setIcon(icon_from_draw(current_draw, size=15, color=icon_color))
        self.fullscreen_btn.setIcon(icon_from_draw(_draw_fullscreen, size=15, color=icon_color))

        self.time_elapsed_label.setStyleSheet(
            f"color: {label_color}; font-size: 12.5px; font-weight: 600; border: none; background: transparent;"
        )
        self.time_total_label.setStyleSheet(
            f"color: {label_color}; font-size: 12.5px; font-weight: 600; border: none; background: transparent;"
        )

    def _center_on_screen(self):
        """Center the window on the primary screen's available (non-taskbar)
        area. Called right after resize() so width/height are already set;
        without this the window manager places the top-level widget at an
        arbitrary offset (observed floating toward the right edge)."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        x = available.x() + (available.width() - self.width()) // 2
        y = available.y() + (available.height() - self.height()) // 2
        self.move(max(available.x(), x), max(available.y(), y))

    # ------------------------------------------------------------
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a video to analyze", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if not path:
            return
        self.start_analysis(path)

    def _refresh_profile_combo(self):
        current = self.profile_combo.currentText() if self.profile_combo.count() else "(default)"
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("(default)")
        for path in find_profiles():
            self.profile_combo.addItem(os.path.basename(path), path)
        idx = self.profile_combo.findText(current)
        self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.profile_combo.blockSignals(False)

    def on_profile_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a camera profile", "camera_profiles", "JSON (*.json)"
        )
        if not path:
            return
        self._use_profile(path)

    def on_calibrate(self):
        dlg = CalibrationDialog(self)
        dlg.profile_saved.connect(self._use_profile)
        dlg.exec()

    def _use_profile(self, path):
        try:
            cfg = load_profile(path)
        except Exception as e:
            QMessageBox.critical(self, "Profile error", str(e))
            return
        self.camera_cfg = cfg
        self.camera_profile_path = path
        idx = self.profile_combo.findText(os.path.basename(path))
        if idx >= 0:
            self.profile_combo.setCurrentIndex(idx)
        else:
            self._refresh_profile_combo()
        n = len(cfg.homography_src_points)
        self._set_status(
            f"Camera profile: {os.path.basename(path)} (camera={cfg.camera_id}, "
            f"homography points={n})",
            "ok",
        )

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

        # Clear previous report cards but keep the empty-state widget (it's
        # re-shown below and hidden again on the first new alert).
        for alert_id, card in list(self.alert_cards.items()):
            self.report_layout.removeWidget(card)
            card.deleteLater()
        self.alert_cards.clear()
        self.alert_count = 0
        self.alert_badge.setText("0")
        self.alert_badge.setStyleSheet(BADGE_STYLE_IDLE)
        self.summary_label.setText("No alerts yet.")
        self.summary_label.setVisible(False)
        self.report_empty_state.setVisible(True)

        px_per_meter = self.calib_spin.value() or None
        self.worker = VideoWorker(
            source_path=path,
            device=self.device_combo.currentText(),
            px_per_meter=px_per_meter,
            min_severity=self.severity_combo.currentText(),
            run_accident=self.accident_check.isChecked(),
            run_violence=self.violence_check.isChecked(),
            camera_cfg=self.camera_cfg,
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
        self.pause_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_ACCENT))
        self._is_playing = True
        self._set_playback_controls_enabled(True)
        self.play_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_SLATE))
        self.stop_btn.setEnabled(True)
        self.upload_btn.setEnabled(False)
        self._set_status(f"Analyzing: {os.path.basename(path)}", "busy")

    def on_pause(self):
        if not self.worker:
            return
        paused = self.worker.toggle_pause()
        self._is_playing = not paused
        if paused:
            self.pause_btn.setText("Resume")
            self.pause_btn.setIcon(icon_from_draw(_draw_play, size=14, color=COLOR_ACCENT))
            self.play_btn.setIcon(icon_from_draw(_draw_play, size=15, color=COLOR_SLATE))
        else:
            self.pause_btn.setText("Pause")
            self.pause_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_ACCENT))
            self.play_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_SLATE))

    def on_open_control_room(self):
        """Launch the control-room console server (if not already running)
        and open it in the default browser. The console is a separate process
        that watches the SAME absolute alerts.jsonl + vista_clips/ this GUI
        writes to — the two-interface demo flow (this GUI detects, the
        console displays with siren/clip). The console is spawned detached
        with an explicit cwd so it works no matter where the GUI itself was
        launched from, and it outlives this window."""
        if not self._control_room_running():
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self.control_proc = subprocess.Popen(
                [sys.executable, "-m", "vista_accident.tools.dashboard",
                 "--log", ALERTS_LOG, "--clips", CLIP_DIR,
                 "--recipients", RECIPIENTS_PATH, "--acks", ACKS_PATH],
                cwd=BASE_DIR,
                creationflags=flags,
            )
        QDesktopServices.openUrl(QUrl(CONTROL_ROOM_URL))

    @staticmethod
    def _control_room_running():
        """True if something already listens on the console port — a stale
        server started earlier would otherwise steal the port and a second
        spawn would die silently, leaving the browser on the wrong instance."""
        import socket
        try:
            with socket.create_connection(("127.0.0.1", 8787), timeout=0.5):
                return True
        except OSError:
            return False

    def on_stop(self):
        if self.worker:
            self.worker.stop()
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.setIcon(icon_from_draw(_draw_pause, size=15, color=COLOR_ACCENT))
        self.stop_btn.setEnabled(False)
        self._is_playing = False
        self._set_playback_controls_enabled(False)
        self.upload_btn.setEnabled(True)
        self._set_status("Stopped.", "idle")

    def on_frame(self, frame):
        if self._video_stack.currentIndex() != 1:
            self._video_stack.setCurrentIndex(1)
        self._last_frame = frame
        pix = bgr_to_qpixmap_fit(frame, self.video_display_wrap.size())
        self.video_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if (
            self._last_frame is not None
            and hasattr(self, "_video_stack")
            and self._video_stack.currentIndex() == 1
            and hasattr(self, "video_display_wrap")
        ):
            pix = bgr_to_qpixmap_fit(
                self._last_frame,
                self.video_display_wrap.size(),
            )
            self.video_label.setPixmap(pix)

    def on_alert(self, payload, shots):
        self.alert_count += 1
        self.report_empty_state.setVisible(False)
        self.summary_label.setVisible(True)
        card = AlertCard(payload, shots)
        self.alert_cards[payload.alert_id] = card
        self.report_layout.insertWidget(0, card)
        self.summary_label.setText(f"{self.alert_count} alert(s) dispatched.")
        self.alert_badge.setText(str(self.alert_count))
        self.alert_badge.setStyleSheet(BADGE_STYLE_ACTIVE)

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
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)

        # Keep the scrub row's elapsed/total timestamps in sync using the
        # worker's real source fps (falls back to 25.0 before the worker
        # has opened the video).
        fps = getattr(self.worker, "fps", 25.0) if self.worker else 25.0
        self.time_elapsed_label.setText(self._fmt_time(frame_idx / fps))
        if total_frames:
            self.time_total_label.setText(self._fmt_time(total_frames / fps))

    @staticmethod
    def _fmt_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"

    def on_finished(self, summary):
        clips = summary.get("clips_saved", 0)
        self._set_status(
            f"Done — {summary['frames']} frames processed, "
            f"{summary['confirmed']} confirmed, {summary['dispatched']} dispatched, "
            f"{clips} clip(s) saved to {CLIP_DIR}.",
            "ok",
        )
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._is_playing = False
        self._set_playback_controls_enabled(False)
        self.upload_btn.setEnabled(True)

    def on_error(self, message):
        QMessageBox.critical(self, "Analysis error", message)
        self._set_status("Error — see dialog.", "error")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._is_playing = False
        self._set_playback_controls_enabled(False)
        self.upload_btn.setEnabled(True)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        # NOTE: the control-room console is deliberately left running — it's
        # an independent display (big-screen demo), and the port-busy check
        # in on_open_control_room prevents duplicate servers.
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_SHEET)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()