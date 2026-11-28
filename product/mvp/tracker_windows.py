"""Interfaz Windows para Tracker D."""

from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from os import getenv
from pathlib import Path
import sys
from threading import Lock
from time import monotonic
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QSettings, QSignalBlocker, QThread, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMainWindow,
                               QFrame, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

MVP = Path(__file__).resolve().parent
sys.path[:0] = [str(MVP), str(MVP.parent / "training/head_control_tracker/v2")]
from calibrate_user import GUIDED_RUNTIME, GUIDED_V2_RUNTIME, RUNTIME, Tracker, primary_size, profile_matrix
from accessibility import AccessBar, DesktopActions, Dwell, DwellRing
from absolute_smoother import (AdaptiveSmoother, ConsensusFixation, ConsensusSmoother, StableFixation,
                               SteadyAdaptiveSmoother, PROFILES as ABSOLUTE_PROFILES)
from desktop_windows import KeyboardDock, WindowsDesktop
from blink_guard import BlinkGuard, blink_limits
from calibration_math import affine
from coverage2d import load_active as load_coverage
from features9 import Camera9, neutral_reference, vector
from reach import MAX_GAIN, active_reach, compare_neutral, set_active
from relative_cursor import DEFAULT_DEADZONE, RelativeCursor, stable_neutral
from validate_cursor import MAX_STEP_PX, PROFILE, SPEED_PX_S, Safety
from win_sink import WindowsSink
from wink_camera import enable_eye_outputs, read_eyes
from wink_clicks import EyeThresholds, GESTURES, LEFT_GESTURES, WinkDetector, WinkSequence, blink_channels

MODEL_OPTIONS = (
    ("Tracker D original", RUNTIME),
    ("Tracker D guiado (experimental)", GUIDED_RUNTIME),
    ("Tracker D guiado v2 (experimental)", GUIDED_V2_RUNTIME),
)
WINK_LOG = Path(getenv("LOCALAPPDATA", str(Path.home()))) / "HeadControlAI" / "wink-events.jsonl"


def xy(point, precision=0):
    return "—" if point is None else f"X {point[0]:.{precision}f}, Y {point[1]:.{precision}f}"


def axes(flags):
    return ", ".join(axis for axis, active in zip("XY", flags) if active) or "ninguno"


def screen_sides(point, scale):
    names = (("izquierdo", "derecho"), ("superior", "inferior"))
    sides = [low if value <= 0 else high for value, maximum, (low, high) in zip(point, scale, names)
             if value <= 0 or value >= maximum]
    return ", ".join(sides) or "interior"


class Vision(QThread):
    status = Signal(str)

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.runtime = RUNTIME
        self.lock = Lock()
        self.latest = None
        self.eyes = None
        self.pose = None
        self.eye_mirrored = True
        self.read_status = "pendiente"
        self.duplicate_frames = 0
        self.matrix = None
        self.reach = None
        self.reach_status = "Sin perfil de alcance"
        self.reach_neutral = None
        self.neutral_prediction = None
        self.coverage = None
        self.coverage_status = "Sin perfil de cobertura 2D"
        self.losses = 0

    def run(self):
        self.reach_neutral = None
        try:
            tracker = Tracker(self.runtime)
            with ExitStack() as stack:
                camera = Camera9(stack)
                self.eye_mirrored = camera.metadata["mirrored"]
                enable_eye_outputs(camera, stack)
                try:
                    self.matrix = profile_matrix(PROFILE, tracker.identity, self.size, camera.metadata)
                except (OSError, KeyError, ValueError):
                    self.matrix = None
                try:
                    self.reach = active_reach(tracker.identity, self.size, camera.metadata)
                    self.reach_status = (
                        f"Alcance compatible · {self.reach.source.parent.name}" if self.reach is not None
                        else "Sin perfil de alcance seleccionado"
                    )
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    self.reach = None
                    self.reach_status = f"Alcance no disponible. {exc}"
                try:
                    self.coverage = load_coverage(tracker.identity, self.size, camera.metadata)
                    self.coverage_status = ("Cobertura 2D disponible" if self.coverage is not None
                                            else "Sin perfil de cobertura 2D")
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    self.coverage = None
                    self.coverage_status = f"Cobertura 2D no disponible. {exc}"
                self.status.emit("Calibrando neutral. Mira al frente durante dos segundos.")
                start, samples, baseline = monotonic(), [], None
                while not self.isInterruptionRequested():
                    now, row, _, landmarks = camera.read()
                    if baseline is None:
                        if now - start >= .5:
                            samples.append(row)
                        if now - start >= 2.5:
                            try:
                                baseline = neutral_reference(samples)
                                neutral = [p for row in samples if row["valid_detection"]
                                           if (p := tracker.predict(row, baseline)) is not None]
                                if self.reach is not None and neutral:
                                    self.reach_neutral = (baseline, np.median(neutral, axis=0))
                                try:
                                    self.neutral_prediction = stable_neutral(neutral)
                                except ValueError:
                                    self.neutral_prediction = None
                                self.status.emit("Neutral listo. Pulsa Activar cursor.")
                            except ValueError:
                                start, samples = now, []
                                self.status.emit("Neutral inestable. Mantén una postura cómoda y quieta.")
                        continue
                    point = tracker.predict(row, baseline) if row["valid_detection"] else None
                    eyes = read_eyes(camera, row, landmarks) if point is not None else None
                    pose = None
                    if point is not None:
                        features = vector(row, baseline)
                        pose = np.array([features[k] for k in ("yaw_rel_deg", "pitch_rel_deg", "roll_rel_deg",
                                                                "face_center_dx", "face_center_dy")]) * [1, 1, .5, 26, 26]
                        if not np.isfinite(pose).all():
                            pose = None
                    self.publish_sample(now, point, eyes, row["status"], pose)
        except Exception as exc:
            self.status.emit(f"Seguimiento detenido. {exc}")
        finally:
            with self.lock:
                self.latest = None
                self.eyes = None
                self.pose = None

    def publish_sample(self, stamp, point, eyes, status, pose=None):
        with self.lock:
            self.read_status = status
            if status == "duplicate_frame":
                self.duplicate_frames += 1
                return
            if point is None:
                self.losses += 1
            self.latest = (stamp, point)
            self.eyes = eyes
            self.pose = pose


DEMO_STYLE = """
QMainWindow, QWidget#demo, QWidget#section { background:#0c1422; }
QWidget { color:#edf4ff; font-family:"Segoe UI"; font-size:14px; }
QFrame#card { background:#172439; border:1px solid #293e5a; border-radius:13px; }
QLabel#title { font-size:30px; font-weight:700; }
QLabel#sectionTitle { font-size:18px; font-weight:650; }
QLabel#muted { color:#afc1da; }
QLabel#state { color:#78b8ff; font-size:16px; font-weight:650; }
QLabel#state[tone="active"] { color:#7ce0af; }
QLabel#state[tone="warning"] { color:#ffd08a; }
QTabWidget::pane, QScrollArea { border:none; background:#0c1422; }
QTabBar::tab { background:#16243a; color:#afc1da; padding:9px 16px; margin-right:4px; border-radius:8px; }
QTabBar::tab:selected { background:#1d4490; color:white; }
QPushButton { min-height:24px; padding:9px 12px; border:1px solid #365576; border-radius:8px; background:#223753; }
QPushButton:hover { background:#29476a; }
QPushButton#primary { background:#235cba; border-color:#488aff; font-weight:650; }
QPushButton#primary:hover { background:#2d6cd2; }
QPushButton:disabled, QComboBox:disabled, QCheckBox:disabled { color:#a1afc3; background:#172439; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { padding:6px 8px; border:1px solid #385476; border-radius:6px; background:#101d30; }
QComboBox QAbstractItemView { background:#16243a; selection-background-color:#1d4490; }
QPushButton:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border:1px solid #78b8ff; }
"""


class Window(QMainWindow):
    def __init__(self, consensus=False, presentation=False):
        super().__init__()
        self.setWindowTitle("HeadControl AI | Tracker D")
        self.setMinimumWidth(360)
        self.resize(570, 655)
        self.demo_was_active = False
        self.demo_settings = (QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                                         "HeadControlAI", "DemostracionWindows") if presentation else None)
        size = primary_size()
        self.scale = np.array(size) - 1
        self.vision = Vision(size)
        self.model_switch_pending = False
        self.safety, self.dwell = Safety(.12), Dwell()
        self.fixation = StableFixation()
        self.actions, self.bar_dwell = DesktopActions(), Dwell()
        self.bar_target = self.bar_fired = None
        self.resume_requires_exit = True
        self.access_bar = AccessBar(self.toolbar_action)
        desktop = WindowsDesktop()
        self.keyboard_dock = KeyboardDock(desktop, QTimer.singleShot, self.report_keyboard)
        self.dwell_ring = DwellRing(desktop.keep_indicator_on_top)
        self.dwell_ring.progress_changed.connect(self.access_bar.show_progress)
        self.wink_detector = WinkDetector()
        self.wink_sequence = WinkSequence()
        self.wink_thresholds = self.eye_calibration = self.eye_window = None
        self.eye_steps = GESTURES
        self.wink_stamp = self.wink_position = None
        self.wink_preview_notice = None
        self.wink_losses = 0
        self.blink_guard = BlinkGuard()
        self.blink_position = None
        self.eye_audit_state = None
        self.eye_status_key = None
        self.winks_effective = False
        self.eye_session = None
        self.relative = RelativeCursor(.12, DEFAULT_DEADZONE, SPEED_PX_S, MAX_STEP_PX)
        self.relative_neutral = None
        self.relative_trace = None
        self.recentering = None
        self.armed = False
        self.face_rearm = False
        self.f8 = False
        self.last_sample = None
        self.armed_losses = 0
        self.last_command = None
        self.control_note = "Sin envío de coordenadas"
        self.api = ctypes.windll.user32
        self.sink = WindowsSink()
        body = QWidget()
        self.setCentralWidget(body)
        body.setObjectName("demo")
        shell = QVBoxLayout(body)
        shell.setContentsMargins(20, 16, 20, 16)
        shell.setSpacing(10)
        title = QLabel("HeadControl AI")
        title.setObjectName("title")
        shell.addWidget(title)
        subtitle = QLabel("Control sin manos · Windows")
        subtitle.setObjectName("muted")
        shell.addWidget(subtitle)
        self.sections = QTabWidget()
        shell.addWidget(self.sections, 1)
        home = self.demo_section("Inicio")
        settings_page = self.demo_section("Ajustes")
        advanced = self.demo_section("Avanzado")
        state_card = self.demo_card(home, "Seguimiento")
        self.home_model = QLabel()
        self.home_model.setWordWrap(True)
        state_card.addWidget(self.home_model)
        self.home_state = QLabel()
        self.home_state.setObjectName("state")
        state_card.addWidget(self.home_state)
        self.home_detail = QLabel()
        self.home_detail.setObjectName("muted")
        self.home_detail.setWordWrap(True)
        state_card.addWidget(self.home_detail)
        home_actions = self.demo_card(home, "Controles")
        layout = self.demo_card(advanced, "Modelos y calibración")
        self.model = QComboBox()
        self.model.addItems([name for name, _ in MODEL_OPTIONS])
        self.model.currentIndexChanged.connect(self.change_model)
        layout.addWidget(self.model)
        self.status = QLabel("Inicia la cámara para calibrar.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.motion = QLabel("Cursor pausado")
        layout.addWidget(self.motion)
        self.control = QComboBox()
        self.control.addItems(("Absoluto", "Relativo experimental"))
        layout.addWidget(self.control)
        self.relative_controls = QWidget()
        relative_layout = QVBoxLayout(self.relative_controls)
        relative_layout.setContentsMargins(0, 0, 0, 0)
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Zona muerta RAW"))
        self.deadzone = QDoubleSpinBox()
        self.deadzone.setDecimals(3)
        self.deadzone.setRange(.005, .100)
        self.deadzone.setSingleStep(.005)
        self.deadzone.setValue(DEFAULT_DEADZONE)
        self.deadzone.valueChanged.connect(self.change_relative_settings)
        settings.addWidget(self.deadzone)
        settings.addWidget(QLabel("Velocidad máxima"))
        self.relative_speed = QSpinBox()
        self.relative_speed.setRange(100, int(SPEED_PX_S))
        self.relative_speed.setSingleStep(50)
        self.relative_speed.setValue(int(SPEED_PX_S))
        self.relative_speed.setSuffix(" px/s")
        self.relative_speed.valueChanged.connect(self.change_relative_settings)
        settings.addWidget(self.relative_speed)
        relative_layout.addLayout(settings)
        self.recenter = QPushButton("Recentrar neutral relativo")
        self.recenter.clicked.connect(self.begin_recenter)
        relative_layout.addWidget(self.recenter)
        self.relative_status = QLabel("Relativo usa el RAW de Tracker D y mueve el cursor desde su posición actual.")
        self.relative_status.setWordWrap(True)
        relative_layout.addWidget(self.relative_status)
        self.relative_controls.hide()
        layout.addWidget(self.relative_controls)
        layout.addWidget(QLabel("Modo de coordenadas"))
        self.mode = QComboBox()
        self.mode.addItems(("RAW", "Afín", "Alcance", "Cobertura 2D"))
        self.mode.model().item(1).setEnabled(False)
        self.mode.model().item(2).setEnabled(False)
        self.mode.model().item(3).setEnabled(False)
        self.mode.currentIndexChanged.connect(self.pause)
        self.control.currentIndexChanged.connect(self.change_control)
        layout.addWidget(self.mode)
        self.reach_status = QLabel(self.vision.reach_status)
        self.reach_status.setWordWrap(True)
        layout.addWidget(self.reach_status)
        self.coverage_status = QLabel(self.vision.coverage_status)
        self.coverage_status.setWordWrap(True)
        layout.addWidget(self.coverage_status)
        self.reset_reach_button = QPushButton("Restablecer alcance")
        self.reset_reach_button.setEnabled(False)
        self.reset_reach_button.clicked.connect(self.reset_reach)
        layout.addWidget(self.reset_reach_button)
        self.clicks = QCheckBox("Clic izquierdo por permanencia")
        self.clicks.toggled.connect(self.change_click_mode)
        home_actions.addWidget(self.clicks)
        layout = self.demo_card(advanced, "Guiños experimentales")
        self.winks = QCheckBox("Clics por uno o dos guiños izquierdos (Absoluto)")
        self.winks.setEnabled(False)
        self.winks.toggled.connect(self.change_click_mode)
        layout.addWidget(self.winks)
        eye_controls = QHBoxLayout()
        self.eye_button = QPushButton("Calibrar ojos")
        self.eye_button.clicked.connect(self.calibrate_eyes)
        eye_controls.addWidget(self.eye_button)
        self.left_eye_button = QPushButton("Recalibrar izquierdo")
        self.left_eye_button.setEnabled(False)
        self.left_eye_button.clicked.connect(lambda: self.calibrate_eyes(left_only=True))
        eye_controls.addWidget(self.left_eye_button)
        self.wink_duration = QSpinBox()
        self.wink_duration.setRange(200, 1500)
        self.wink_duration.setValue(300)
        self.wink_duration.setSuffix(" ms")
        self.wink_duration.valueChanged.connect(self.configure_winks)
        eye_controls.addWidget(QLabel("Guiño mínimo"))
        eye_controls.addWidget(self.wink_duration)
        self.wink_cooldown = QSpinBox()
        self.wink_cooldown.setRange(300, 2500)
        self.wink_cooldown.setValue(650)
        self.wink_cooldown.setSuffix(" ms")
        self.wink_cooldown.valueChanged.connect(self.configure_winks)
        eye_controls.addWidget(QLabel("Enfriamiento"))
        eye_controls.addWidget(self.wink_cooldown)
        layout.addLayout(eye_controls)
        blink_controls = QHBoxLayout()
        blink_controls.addWidget(QLabel("Ventana de doble guiño"))
        self.wink_interval = QSpinBox()
        self.wink_interval.setRange(350, 1500)
        self.wink_interval.setValue(600)
        self.wink_interval.setSuffix(" ms")
        self.wink_interval.valueChanged.connect(self.configure_winks)
        blink_controls.addWidget(self.wink_interval)
        blink_controls.addWidget(QLabel("Espera tras reapertura"))
        self.blink_recovery = QSpinBox()
        self.blink_recovery.setRange(100, 500)
        self.blink_recovery.setValue(180)
        self.blink_recovery.setSuffix(" ms")
        self.blink_recovery.valueChanged.connect(self.configure_blinks)
        blink_controls.addWidget(self.blink_recovery)
        self.wink_log = QCheckBox("Registrar diagnóstico ocular")
        self.wink_log.setToolTip(str(WINK_LOG))
        self.wink_log.toggled.connect(self.eye_logging_changed)
        blink_controls.addWidget(self.wink_log)
        layout.addLayout(blink_controls)
        self.blink_label = QLabel("Protección ocular · ojos abiertos")
        layout.addWidget(self.blink_label)
        self.wink_status = QLabel("Cursor pausado · guiños desactivados")
        self.wink_status.setWordWrap(True)
        layout.addWidget(self.wink_status)
        self.eye_preview = QLabel("Ojos · inicia la cámara para ver la señal.")
        self.eye_preview.setWordWrap(True)
        layout.addWidget(self.eye_preview)
        self.eye_hint = QLabel("Calibra los ojos para probar los gestos sin emitir clics.")
        self.eye_hint.setWordWrap(True)
        layout.addWidget(self.eye_hint)
        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 500)
        self.smoothing.setValue(120)
        self.smoothing.setSuffix(" ms")
        self.smoothing.valueChanged.connect(self.change_smoothing)
        layout.addWidget(QLabel("Suavizado Afín/Alcance/Cobertura 2D y Relativo"))
        layout.addWidget(self.smoothing)
        layout = self.demo_card(settings_page, "Respuesta del puntero")
        absolute_controls = QHBoxLayout()
        absolute_controls.addWidget(QLabel("Estabilización Absoluto / RAW"))
        self.absolute_strength = QComboBox()
        self.absolute_strength.addItems(("Desactivado", *ABSOLUTE_PROFILES,
                                         *(f"{name} anterior" for name in ABSOLUTE_PROFILES)))
        self.absolute_strength.setCurrentText("Suave")
        self.absolute_strength.currentIndexChanged.connect(self.change_absolute_settings)
        absolute_controls.addWidget(self.absolute_strength)
        absolute_controls.addWidget(QLabel("Zona muerta"))
        self.absolute_deadzone = QDoubleSpinBox()
        self.absolute_deadzone.setRange(0., 6.)
        self.absolute_deadzone.setSingleStep(.5)
        self.absolute_deadzone.setValue(1.5)
        self.absolute_deadzone.setSuffix(" px")
        self.absolute_deadzone.valueChanged.connect(self.change_absolute_settings)
        absolute_controls.addWidget(self.absolute_deadzone)
        layout.addLayout(absolute_controls)
        stable_controls = QHBoxLayout()
        stable_controls.addWidget(QLabel("Radio de estabilidad RAW"))
        self.absolute_stability = QDoubleSpinBox()
        self.absolute_stability.setRange(0., 10.)
        self.absolute_stability.setSingleStep(.5)
        self.absolute_stability.setValue(5.)
        self.absolute_stability.setSuffix(" px")
        self.absolute_stability.valueChanged.connect(self.change_absolute_settings)
        stable_controls.addWidget(self.absolute_stability)
        layout.addLayout(stable_controls)
        fixation_controls = QHBoxLayout()
        self.fixation_enabled = QCheckBox("Fijación estable")
        self.fixation_enabled.setChecked(True)
        self.fixation_enabled.toggled.connect(self.change_absolute_settings)
        fixation_controls.addWidget(self.fixation_enabled)
        fixation_controls.addWidget(QLabel("Salida"))
        self.fixation_exit = QSpinBox()
        self.fixation_exit.setRange(12, 45)
        self.fixation_exit.setValue(20)
        self.fixation_exit.setSuffix(" px")
        self.fixation_exit.valueChanged.connect(self.change_absolute_settings)
        fixation_controls.addWidget(self.fixation_exit)
        self.fixation_confirm = QSpinBox()
        self.fixation_confirm.setRange(30, 180)
        self.fixation_confirm.setValue(60)
        self.fixation_confirm.setSuffix(" ms")
        self.fixation_confirm.valueChanged.connect(self.change_absolute_settings)
        fixation_controls.addWidget(self.fixation_confirm)
        self.fixation_drag = QCheckBox("Anclar al arrastrar")
        self.fixation_drag.toggled.connect(self.change_absolute_settings)
        fixation_controls.addWidget(self.fixation_drag)
        layout.addLayout(fixation_controls)
        self.fixation_method = QComboBox()
        self.fixation_method.addItems(("Fijación estable original", "Estabilización por consenso"))
        self.fixation_method.setCurrentIndex(int(consensus))
        self.fixation_method.currentIndexChanged.connect(self.change_absolute_settings)
        layout.addWidget(self.fixation_method)
        self.fixation_status = QLabel()
        layout.addWidget(self.fixation_status)
        self.mode.currentIndexChanged.connect(self.configure_absolute)
        self.restore_demo_settings()
        self.configure_absolute()
        actions = self.demo_card(settings_page, "Acciones de Windows")
        keyboard = QPushButton("Abrir teclado en pantalla")
        keyboard.clicked.connect(lambda: self.keyboard_dock.open(self.sink.keyboard))
        actions.addWidget(keyboard)
        tips = QLabel("Usa la barra de control para arrastrar o desplazar.")
        tips.setWordWrap(True)
        actions.addWidget(tips)
        layout = home_actions
        self.start_camera = QPushButton("Iniciar cámara y calibrar")
        self.start_camera.clicked.connect(self.start)
        self.start_camera.setObjectName("primary")
        layout.insertWidget(1, self.start_camera)
        self.arm = QPushButton("Activar cursor")
        self.arm.clicked.connect(self.toggle)
        layout.insertWidget(2, self.arm)
        layout.addWidget(QLabel("F8 o Esc pausa. Al perder el rostro, vuelve a pulsar Activar."))
        layout = self.demo_card(advanced, "Diagnóstico")
        self.diagnostic_toggle = QCheckBox("Mostrar diagnóstico del cursor")
        layout.addWidget(self.diagnostic_toggle)
        self.diagnostic = QPlainTextEdit()
        self.diagnostic.setReadOnly(True)
        self.diagnostic.setMinimumHeight(240)
        self.diagnostic.setVisible(False)
        self.diagnostic_toggle.toggled.connect(self.diagnostic.setVisible)
        layout.addWidget(self.diagnostic)
        for page in (home, settings_page, advanced):
            page.addStretch()
        self.setStyleSheet(DEMO_STYLE)
        self.refresh_demo_state()
        self.vision.status.connect(self.vision_status)
        self.vision.finished.connect(self.finished)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

    def demo_section(self, name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName("section")
        scroll.setWidget(page)
        self.sections.addTab(scroll, name)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(2, 6, 8, 6)
        layout.setSpacing(10)
        return layout

    @staticmethod
    def demo_card(page, title):
        card = QFrame()
        card.setObjectName("card")
        page.addWidget(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return layout

    def demo_controls(self):
        return {"filter": self.absolute_strength, "method": self.fixation_method,
                "deadzone": self.absolute_deadzone, "stability": self.absolute_stability,
                "exit": self.fixation_exit, "confirmation": self.fixation_confirm,
                "fixation": self.fixation_enabled, "drag_fixation": self.fixation_drag}

    def restore_demo_settings(self):
        if self.demo_settings is None:
            return
        for key, field in self.demo_controls().items():
            with QSignalBlocker(field):
                if isinstance(field, QComboBox):
                    field.setCurrentText(self.demo_settings.value(key, field.currentText(), type=str))
                elif isinstance(field, QCheckBox):
                    field.setChecked(self.demo_settings.value(key, field.isChecked(), type=bool))
                else:
                    field.setValue(self.demo_settings.value(key, field.value(), type=type(field.value())))

    def save_demo_settings(self):
        if self.demo_settings is None:
            return
        for key, field in self.demo_controls().items():
            value = field.currentText() if isinstance(field, QComboBox) else (
                field.isChecked() if isinstance(field, QCheckBox) else field.value())
            self.demo_settings.setValue(key, value)
        self.demo_settings.sync()

    def refresh_demo_state(self):
        self.home_model.setText(f"{self.model.currentText().split(' (')[0]} · {self.control.currentText()} / {self.mode.currentText()}\n"
                                f"{self.fixation_method.currentText()}")
        if not self.vision.isRunning():
            self.demo_was_active = False
        if self.armed:
            self.demo_was_active = True
            state, tone, detail = "Activo", "active", "Cámara activa · neutral listo · puntero activo"
        elif self.vision.isRunning() and self.vision.neutral_prediction is not None:
            paused = self.demo_was_active or self.face_rearm
            state = "Pausado" if paused else "Listo"
            tone, detail = "warning" if paused else "", "Cámara activa · neutral listo · puntero detenido"
        elif self.vision.isRunning():
            state, tone, detail = "Calibrando", "warning", "Cámara en preparación · neutral pendiente"
        elif self.status.text().startswith("Seguimiento detenido."):
            state, tone, detail = "Error", "warning", "Revisa Avanzado para ver el detalle."
        else:
            state, tone, detail = "Inactivo", "", "Cámara detenida · neutral pendiente"
        if self.home_state.text() != state:
            self.home_state.setText(state)
            self.home_state.setProperty("tone", tone)
            self.home_state.style().unpolish(self.home_state)
            self.home_state.style().polish(self.home_state)
        self.home_detail.setText(detail)

    def start(self):
        if self.start_camera.isEnabled() and not self.model_switch_pending and not self.vision.isRunning():
            _, self.vision.runtime = MODEL_OPTIONS[self.model.currentIndex()]
            self.vision.neutral_prediction = self.relative_neutral = None
            self.start_camera.setEnabled(False)
            self.vision.start()
            self.access_bar.place()
            self.access_bar.show()

    def change_model(self, index):
        self.pause()
        self.clicks.setChecked(False)
        self.winks.setChecked(False)
        self.last_sample = self.last_command = self.relative_neutral = None
        with self.vision.lock:
            self.vision.latest = None
        self.clear_profile_controls()
        if not self.start_camera.isEnabled() or self.vision.isRunning():
            self.model_switch_pending = True
            self.vision.requestInterruption()
            self.status.setText("Cerrando la cámara del modelo anterior…")
        else:
            self.vision.neutral_prediction = None
            self.prompt_model_neutral()

    def prompt_model_neutral(self):
        self.status.setText(f"{self.model.currentText()} seleccionado. Pulsa Iniciar cámara y calibrar "
                            "para medir un neutral nuevo.")

    def vision_status(self, text):
        if not self.model_switch_pending:
            self.status.setText(text)

    def change_smoothing(self, milliseconds):
        self.pause()
        self.configure_absolute()
        self.configure_relative()

    def configure_absolute(self):
        raw = self.mode.currentIndex() == 0
        self.safety = Safety(0. if raw else self.smoothing.value() / 1000.)
        selected = self.absolute_strength.currentText()
        consensus = self.fixation_method.currentIndex() == 1
        fixation_class = ConsensusFixation if consensus else StableFixation
        self.fixation = fixation_class(self.fixation_exit.value(), self.fixation_confirm.value() / 1000.)
        self.update_fixation_label()
        profile = selected.removesuffix(" anterior")
        if raw and profile in ABSOLUTE_PROFILES:
            deadzone = self.absolute_deadzone.value()
            if selected == profile:
                smoother = ConsensusSmoother if consensus else SteadyAdaptiveSmoother
                self.safety.smooth = smoother(self.scale, profile, deadzone, self.absolute_stability.value())
            else:
                self.safety.smooth = AdaptiveSmoother(self.scale, profile, deadzone)

    def change_absolute_settings(self):
        self.pause()
        self.configure_absolute()
        self.save_demo_settings()

    def fixation_active(self):
        return (self.fixation_enabled.isChecked() and self.absolute_strength.currentText() in ABSOLUTE_PROFILES
                and self.mode.currentIndex() == self.control.currentIndex() == 0
                and (not self.actions.dragging or self.fixation_drag.isChecked()))

    def update_fixation_label(self):
        state = self.fixation.state if self.fixation_active() else "desactivada"
        message = f"{self.fixation_method.currentText()} · {state}"
        if self.fixation_status.text() != message:
            self.fixation_status.setText(message)

    def report_keyboard(self, message):
        self.status.setText(message)

    def configure_relative(self):
        self.relative = RelativeCursor(self.smoothing.value() / 1000., self.deadzone.value(),
                                       self.relative_speed.value(), MAX_STEP_PX)

    def change_click_mode(self, enabled):
        self.pause()
        if enabled:
            other = self.winks if self.sender() is self.clicks else self.clicks
            other.setChecked(False)

    def change_desktop_clicks(self, enabled):
        if enabled:
            with QSignalBlocker(self.winks):
                self.winks.setChecked(False)
            self.wink_sequence.reset()
            self.wink_detector.reset()
            self.wink_position = None
        with QSignalBlocker(self.clicks):
            self.clicks.setChecked(enabled)
        self.dwell.reset()
        self.refresh_wink_status()

    def release_drag(self):
        try:
            self.actions.release(self.sink)
        except Exception as exc:
            self.status.setText(f"Windows rechazó soltar el botón. Reintentando. {exc}")
            return False
        return True

    def access_sample_valid(self, sample, eyes, now):
        if (sample is None or sample[1] is None or now - sample[0] > .25 or eyes is None
                or not np.isfinite(eyes[1]).all()):
            return False
        onset, _ = blink_limits(eyes, self.wink_thresholds)
        return all(value < limit for value, limit in zip(eyes[1], onset))

    def access_ready(self, now):
        with self.vision.lock:
            sample, eyes, losses = self.vision.latest, self.vision.eyes, self.vision.losses
        return (not self.model_switch_pending and self.eye_calibration is None and self.recentering is None
                and not self.stop_pressed() and not self.blink_guard.status.holding
                and self.access_sample_valid(sample, eyes, now)
                and (not self.armed or losses == self.armed_losses))

    def toolbar_action(self, action):
        if action == "pause" and self.armed:
            self.pause()
            return
        if not self.access_ready(monotonic()):
            return
        if action == "pause":
            if not self.resume_requires_exit and not self.actions.dragging:
                self.toggle()
            return
        if not self.armed:
            return
        self.dwell.reset()
        try:
            if action == "clicks":
                self.actions.select("left", self.sink)
                self.change_desktop_clicks(not self.clicks.isChecked())
            elif action == "keyboard":
                self.actions.select("left", self.sink)
                self.keyboard_dock.open(self.sink.keyboard)
                self.access_bar.place(keyboard=True)
            elif action == "drag" and self.actions.dragging:
                self.actions.select("left", self.sink)
            else:
                mode = "left" if action == self.actions.mode == "scroll" else action
                self.actions.select(mode, self.sink)
                self.change_desktop_clicks(True)
        except Exception as exc:
            self.pause()
            self.status.setText(f"Windows no pudo completar la acción. {exc}")

    def update_accessibility(self, sample, eyes, losses, now):
        if not self.armed and self.actions.dragging:
            self.release_drag()
        point = self.access_bar.cursor_point()
        target, over_bar = None, False
        bar_progress = False
        valid = (not self.model_switch_pending and self.eye_calibration is None and self.recentering is None
                 and not self.stop_pressed() and not self.blink_guard.status.holding and self.wink_position is None
                 and self.access_sample_valid(sample, eyes, now)
                 and (not self.armed or losses == self.armed_losses))
        if self.access_bar.isVisible():
            if valid:
                if not self.armed:
                    point = self.access_bar.virtual_point(sample[1])
                    over_bar = self.access_bar.contains(point)
                    target = self.access_bar.target(point)
                    if target != "pause":
                        self.resume_requires_exit = False
                        target = None
                    elif self.resume_requires_exit or self.actions.dragging:
                        target = None
                else:
                    over_bar = self.access_bar.contains(point)
                    target = self.access_bar.target(point)
                if target != self.bar_target:
                    self.bar_dwell.reset()
                    self.bar_target = target
                    if target != self.bar_fired:
                        self.bar_fired = None
                if target is not None and self.bar_fired != target:
                    if self.bar_dwell.update(np.array([point.x(), point.y()]), now):
                        self.bar_fired = target
                        self.toolbar_action(target)
                    else:
                        self.dwell_ring.track(point, self.bar_dwell.progress(now))
                        bar_progress = True
            else:
                self.bar_dwell.reset()
                over_bar = self.access_bar.contains(point)
            self.access_bar.render(self.armed, self.clicks.isChecked(), self.actions, target)
        if not valid or over_bar or not self.armed or not self.clicks.isChecked():
            self.dwell.suspend()
            self.actions.stop_scroll()
            if not bar_progress:
                self.dwell_ring.hide()
            return
        if self.control.currentIndex() == 0:
            ready = self.safety.last_sent is not None or self.fixation.anchor is not None
        else:
            ready = self.relative_trace is not None and self.relative_trace.state == "neutral"
        if not ready:
            self.dwell.suspend()
            self.actions.stop_scroll()
            self.dwell_ring.hide()
            return
        position = np.asarray(self.sink.position())
        if np.any((position < 0) | (position > self.scale)):
            self.dwell.reset()
            self.actions.stop_scroll()
            self.dwell_ring.hide()
            return
        try:
            if self.actions.mode == "scroll":
                self.dwell_ring.hide()
                zone = self.access_bar.scroll_zone(point) if self.access_ready(now) else 0
                self.actions.scroll(zone, now, self.sink)
            elif self.dwell.update(position, now):
                self.dwell_ring.hide()
                if self.access_ready(now) and not self.access_bar.contains(self.access_bar.cursor_point()):
                    self.actions.activate(self.sink)
            elif not self.dwell.latched:
                self.dwell_ring.track(point, self.dwell.progress(now))
            else:
                self.dwell_ring.hide()
        except Exception as exc:
            self.pause()
            self.status.setText(f"Windows no pudo completar la acción. {exc}")

    def configure_winks(self):
        self.pause()
        self.wink_detector = WinkDetector(self.wink_duration.value() / 1000.)
        self.wink_sequence = WinkSequence(self.wink_interval.value() / 1000., self.wink_cooldown.value() / 1000.)

    def configure_blinks(self, milliseconds):
        self.pause()
        self.blink_guard.recovery = milliseconds / 1000.

    def wink_blocker(self, sample, reading, losses, now):
        if self.eye_calibration is not None:
            return "Calibración ocular en curso"
        if self.wink_thresholds is None:
            return "Calibra los ojos"
        if not self.winks.isChecked():
            return "Marca la casilla de guiños"
        if self.control.currentIndex() != 0:
            return "Selecciona Absoluto"
        if self.model_switch_pending:
            return "Esperando cambio de modelo"
        if not self.armed:
            return "Pulsa Activar cursor"
        if (sample is None or sample[1] is None or now - sample[0] > .25 or losses != self.armed_losses
                or reading is None or reading[0] != self.wink_thresholds.source):
            return "Esperando detección facial y ocular válida"
        return None

    def refresh_wink_status(self):
        with self.vision.lock:
            sample, reading, losses = self.vision.latest, self.vision.eyes, self.vision.losses
        reason = self.wink_blocker(sample, reading, losses, monotonic())
        self.winks_effective = reason is None
        state = (self.armed, self.winks.isChecked(), reason)
        if state != self.eye_status_key:
            self.eye_status_key = state
            message = (f"Cursor {'armado' if self.armed else 'pausado'} · "
                       f"Guiños {'marcados' if self.winks.isChecked() else 'desmarcados'} · "
                       + ("Clics por guiño habilitados" if reason is None else f"Solo vista previa · {reason}"))
            self.wink_status.setText(message)
            self.log_eye_event("estado controlador", message, reading, sample[0] if sample is not None else None)

    def eye_logging_changed(self, enabled):
        if enabled:
            self.eye_session = uuid4().hex
            self.eye_status_key = None
            self.refresh_wink_status()

    def log_eye_event(self, stage, message, reading=None, stamp=None, **details):
        if not self.wink_log.isChecked():
            return
        data = dict(time_utc=datetime.now(timezone.utc).isoformat(), session_id=self.eye_session,
                    stage=stage, outcome=message, frame_stamp=stamp, armed=self.armed,
                    winks_enabled=self.winks.isChecked(), winks_effective=self.winks_effective,
                    vision_status=self.vision.read_status, duplicate_frames=self.vision.duplicate_frames,
                    min_wink_ms=self.wink_duration.value(), double_wink_ms=self.wink_interval.value(), **details)
        if reading is not None:
            source, values = reading
            data.update(source=source, anatomical=dict(zip(("left", "right"), values)))
            if source == "blendshapes":
                data["raw"] = dict(zip(("eyeBlinkLeft", "eyeBlinkRight"),
                                       blink_channels(values, self.vision.eye_mirrored)))
            if self.wink_thresholds is not None and self.wink_thresholds.source == source:
                data["normalized"] = self.wink_thresholds.normalize(values)
                data["thresholds"] = dict(reopen=self.wink_thresholds.opened, close=self.wink_thresholds.closed,
                                          right_during_left=self.wink_thresholds.right_during_left)
        try:
            WINK_LOG.parent.mkdir(parents=True, exist_ok=True)
            with WINK_LOG.open("a", encoding="utf-8") as log:
                log.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            self.wink_log.setChecked(False)
            self.eye_hint.setText(f"No se pudo registrar el diagnóstico: {exc}")

    def protect_blink(self, sample, reading, losses, now):
        previous = self.blink_guard.status
        valid = sample is not None and sample[1] is not None and now - sample[0] <= .25
        stamp = sample[0] if valid else None
        status = self.blink_guard.update(stamp, reading if valid else None, self.wink_thresholds)
        if status.state != previous.state:
            self.blink_label.setText(f"Protección ocular · {status.state}")
            self.log_eye_event("detección", status.state, reading if valid else None, stamp)
        if not (self.armed and self.control.currentIndex() == 0 and valid and losses == self.armed_losses):
            self.blink_position = None
            return False
        if status.holding:
            if self.blink_position is None:
                self.blink_position = tuple(self.sink.position())
                self.fixation.interrupt()
            self.dwell.suspend()
            return True
        if status.resumed and self.blink_position is not None:
            self.blink_position = None
            self.safety.resume_after_reopen(now)
        return False

    def calibrate_eyes(self, left_only=False):
        now = monotonic()
        with self.vision.lock:
            sample, reading = self.vision.latest, self.vision.eyes
        if self.model_switch_pending or self.stop_pressed() or reading is None or sample is None or now - sample[0] > .25:
            self.eye_hint.setText("Inicia la cámara y espera una lectura ocular válida.")
            return
        if self.eye_calibration is None:
            if left_only and self.wink_thresholds is None:
                return
            self.winks.setChecked(False)
            self.pause()
            self.eye_steps = LEFT_GESTURES if left_only else GESTURES
            self.eye_calibration = []
            self.wink_preview_notice = None
            self.left_eye_button.setEnabled(False)
            self.eye_hint.setText(f"{self.eye_steps[0]}. Pulsa Medir cuando estés listo.")
            self.eye_button.setText("Medir")
        elif self.eye_window is None:
            self.eye_window = (now + .4, now + 1.6, [])
            self.eye_button.setEnabled(False)
            self.eye_hint.setText(f"{self.eye_steps[len(self.eye_calibration)]} durante la medición.")

    def update_eye_calibration(self, stamp, reading, now):
        if self.eye_window is None:
            return
        start, end, readings = self.eye_window
        if reading is not None and stamp is not None and start <= stamp <= end:
            readings.append(reading)
        if now < end:
            return
        self.eye_window = None
        self.eye_button.setEnabled(True)
        if len(readings) < 12:
            self.eye_hint.setText("Lecturas insuficientes. Repite esta postura con Medir.")
            return
        self.eye_calibration.append(readings)
        phase = len(self.eye_calibration)
        if phase < len(self.eye_steps):
            instruction = " y mantén el derecho abierto" if phase == 1 else ""
            self.eye_hint.setText(f"{self.eye_steps[phase]}{instruction}. Pulsa Medir.")
            return
        try:
            if self.eye_steps == LEFT_GESTURES:
                self.wink_thresholds = self.wink_thresholds.recalibrate_left(self.eye_calibration)
            else:
                self.wink_thresholds = EyeThresholds.fit(self.eye_calibration)
        except ValueError as exc:
            self.eye_hint.setText(str(exc))
        else:
            self.wink_detector.reset()
            self.wink_sequence.reset()
            self.winks.setEnabled(True)
            limits = " · ".join(f"{side} {op:.2f}/{cl:.2f}" for side, op, cl in zip(
                ("I", "D"), self.wink_thresholds.opened, self.wink_thresholds.closed))
            self.eye_hint.setText(f"Umbrales abrir/cerrar · {limits}. Prueba en vista previa antes de activar.")
        self.eye_calibration = None
        self.eye_button.setText("Calibrar ojos")
        self.left_eye_button.setEnabled(self.wink_thresholds is not None)

    def show_eye_preview(self, reading, result, sequence, now):
        notice = sequence.state if sequence.button else sequence.notice or result.notice
        if notice:
            self.wink_preview_notice = (now + 2.5, notice)
        lines = []
        if reading is None:
            lines.append("Ojos · sin lectura válida")
        else:
            source, values = reading
            if source == "blendshapes":
                mirrored = self.vision.eye_mirrored
                left, right = blink_channels(values, mirrored)
                lines.append(f"RAW eyeBlinkLeft {left:.3f} → {'D' if mirrored else 'I'} anatómico · "
                             f"eyeBlinkRight {right:.3f} → {'I' if mirrored else 'D'} anatómico")
            else:
                lines.append("EAR · eyeBlinkLeft/Right no disponibles")
            limits = self.wink_thresholds
            normalized = limits.normalize(values) if limits is not None and source == limits.source else None
            for i, name in enumerate(("Izquierdo", "Derecho")):
                line = f"{name} {values[i]:.3f} · {result.eyes[i]}"
                if normalized is not None:
                    line += (f" · norm. {normalized[i]:+.2f} · "
                             f"cierre {limits.closed[i]:.3f} / reapertura {limits.opened[i]:.3f}")
                lines.append(line)
            if normalized is not None:
                lines.append(f"Guiño I · límite del ojo derecho {limits.right_during_left:.3f}")
        notice = self.wink_preview_notice
        if result.state != sequence.state:
            lines.append(result.state)
        lines.append(sequence.state + (f" · {notice[1]}" if notice is not None and now < notice[0]
                                       and notice[1] != sequence.state else ""))
        self.eye_preview.setText("\n".join(lines))

    def update_winks(self, sample, reading, losses, now):
        fresh = sample is not None and sample[1] is not None and now - sample[0] <= .25
        stamp = sample[0] if fresh else None
        if not fresh:
            reading = None
        if stamp is not None and stamp == self.wink_stamp and losses == self.wink_losses:
            self.update_eye_calibration(None, None, now)
            return self.wink_position is not None
        # Un salto en el contador invalida un guiño aunque la UI no viera el fotograma perdido.
        if losses != self.wink_losses:
            if self.wink_detector.start is not None or self.wink_sequence.deadline is not None:
                self.wink_preview_notice = (now + 2.5, "Pérdida facial durante el guiño.")
                self.log_eye_event("clasificación", "Guiño cancelado por pérdida facial.", reading, stamp)
            self.wink_detector.reset()
            self.wink_sequence.reset()
            self.wink_position = None
            self.wink_losses = losses
        self.wink_stamp = stamp
        self.update_eye_calibration(stamp, reading, now)
        result = self.wink_detector.update(stamp, reading, self.wink_thresholds)
        sequence = self.wink_sequence.update(stamp, result)
        self.show_eye_preview(reading, result, sequence, now)
        classification = sequence.notice or result.notice or (
            "Guiño unilateral válido" if result.qualified else "Candidato unilateral" if result.holding else None)
        if result.confirmed:
            self.log_eye_event("confirmación", "Guiño izquierdo confirmado al reabrir.", reading, stamp)
            if sequence.state.startswith("Guiño 1") or sequence.button == "right":
                self.log_eye_event("secuencia", "Segundo guiño registrado." if sequence.button else
                                   "Primer guiño registrado.", reading, stamp)
        elif classification is not None and classification != self.eye_audit_state:
            self.log_eye_event("clasificación", classification, reading, stamp,
                               prior_open_ms=round(result.prior_open_ms, 1))
        self.eye_audit_state = classification
        active = self.wink_blocker(sample, reading, losses, now) is None
        previous, self.wink_position = self.wink_position, None
        if not active:
            if sequence.button:
                self.wink_preview_notice = (now + 2.5, f"{sequence.state} previsto · sin actuación")
                self.log_eye_event("emisión Windows", f"{sequence.state}: solo vista previa.", reading, stamp)
            return False
        if result.holding or self.wink_sequence.deadline is not None:
            self.wink_position = previous if previous is not None else (
                self.blink_position if self.blink_position is not None else tuple(self.sink.position()))
            self.dwell.reset()
        if previous is not None and self.wink_position is None:
            if self.blink_position is None:
                self.safety.resume_after_reopen(now)
            if sequence.button:
                if tuple(self.sink.position()) == previous and all(
                        0 <= value <= bound for value, bound in zip(previous, self.scale)):
                    self.emit_wink(sequence.button, stamp, reading)
                else:
                    self.wink_preview_notice = (now + 2.5, "Gesto reconocido. Clic cancelado por posición del puntero.")
                    self.log_eye_event("emisión Windows", "Cancelado: posición del puntero cambió o está fuera.", reading, stamp)
        elif sequence.button:
            self.log_eye_event("emisión Windows", "Cancelado: faltó una posición congelada.", reading, stamp)
        if (self.wink_position is not None or previous is not None) and fresh:
            self.last_sample = stamp
            return True
        return False

    def emit_wink(self, button, stamp, reading):
        if self.access_bar.contains(self.access_bar.cursor_point()):
            self.log_eye_event("emisión Windows", "Cancelado: puntero sobre la barra.", reading, stamp)
            return
        with self.vision.lock:
            eyes = self.vision.eyes
            latest = self.vision.latest
            valid = (self.vision.losses == self.armed_losses and latest is not None
                     and self.vision.latest[1] is not None and eyes is not None
                     and eyes[0] == self.wink_thresholds.source
                     and 0 <= latest[0] - stamp <= .12 and monotonic() - latest[0] <= .25)
        # La reapertura ya se confirmó en el fotograma que originó el gesto.
        # Una nueva transición leve no invalida esa confirmación; un nuevo cierre sí la cancela.
        if valid and latest[0] > stamp:
            onset, _ = blink_limits(eyes, self.wink_thresholds)
            valid = all(score < trigger for score, trigger in zip(eyes[1], onset))
        if not self.winks_effective or self.stop_pressed() or not valid:
            self.pause()
            self.wink_preview_notice = (monotonic() + 2.5, "Clic cancelado por pausa, pérdida o nuevo cierre.")
            self.log_eye_event("emisión Windows", f"{button}: cancelado por pausa, pérdida o nuevo cierre.", reading, stamp)
            return
        try:
            self.sink.click(button)
        except Exception as exc:
            self.pause()
            self.wink_preview_notice = (monotonic() + 2.5, f"Windows rechazó el clic: {exc}")
            self.log_eye_event("emisión Windows", f"{button}: {type(exc).__name__}: {exc}", reading, stamp)
        else:
            self.wink_preview_notice = (monotonic() + 2.5, "Clic emitido a Windows.")
            self.log_eye_event("emisión Windows", f"{button}: clic enviado.", reading, stamp)

    def change_relative_settings(self):
        self.pause()
        self.configure_relative()

    def change_control(self, index):
        self.pause()
        self.mode.setEnabled(index == 0)
        self.relative_controls.setVisible(index == 1)

    def current_neutral(self):
        return self.vision.neutral_prediction if self.relative_neutral is None else self.relative_neutral

    def begin_recenter(self):
        self.pause()
        if self.model_switch_pending or self.stop_pressed():
            return
        with self.vision.lock:
            sample, losses = self.vision.latest, self.vision.losses
        now = monotonic()
        if sample is None or sample[1] is None or now - sample[0] > .25:
            self.relative_status.setText("Espera a que la cámara detecte el rostro.")
            return
        self.recentering = (now, losses, [])
        self.arm.setEnabled(False)
        self.relative_status.setText("Mantén la cabeza en reposo. Capturando neutral durante 1,8 s.")

    def update_recenter(self, sample, losses, now):
        start, initial_losses, samples = self.recentering
        if losses != initial_losses or sample is None or sample[1] is None or now - sample[0] > .25:
            self.pause()
            self.relative_status.setText("Neutral conservado. Se perdió la detección durante el recentrado.")
            return
        stamp, raw = sample
        if stamp >= start + .6 and (not samples or stamp > samples[-1][0]):
            samples.append((stamp, raw))
        if now >= start + 1.8:
            self.pause()
            try:
                self.relative_neutral = stable_neutral([point for _, point in samples], self.deadzone.value())
            except ValueError as exc:
                self.relative_status.setText(f"Neutral conservado. {exc}")
            else:
                self.relative_status.setText(f"Neutral relativo listo · {xy(self.relative_neutral, 4)}. Pulsa Activar.")

    def finished(self):
        self.pause()
        self.vision.neutral_prediction = self.relative_neutral = None
        self.clear_profile_controls()
        if self.model_switch_pending:
            self.model_switch_pending = False
            self.prompt_model_neutral()
        self.start_camera.setEnabled(True)

    def clear_profile_controls(self):
        self.mode.setCurrentIndex(0)
        self.mode.model().item(1).setEnabled(False)
        self.mode.model().item(2).setEnabled(False)
        self.mode.model().item(3).setEnabled(False)
        self.reset_reach_button.setEnabled(False)
        self.reach_status.setText("Alcance pendiente de comprobar compatibilidad.")
        self.coverage_status.setText("Cobertura 2D pendiente de comprobar compatibilidad.")

    def reset_reach(self):
        self.mode.setCurrentIndex(0)
        self.pause()
        self.vision.reach = None
        self.vision.reach_neutral = None
        try:
            set_active(None)
        except OSError as exc:
            self.vision.reach_status = f"Alcance desactivado esta sesión. Error al guardar: {exc}"
        else:
            self.vision.reach_status = "Alcance restablecido. Perfil anterior conservado."
        self.mode.model().item(2).setEnabled(False)
        self.reset_reach_button.setEnabled(False)
        self.reach_status.setText(self.vision.reach_status)

    def pause(self):
        if self.wink_detector.start is not None or self.wink_sequence.deadline is not None:
            self.log_eye_event("clasificación", "Guiño cancelado por pausa.")
        self.armed = False
        self.release_drag()
        self.actions.mode = "left"
        self.resume_requires_exit = True
        self.bar_dwell.reset()
        self.dwell_ring.hide()
        self.safety.reset()
        self.fixation.reset()
        self.update_fixation_label()
        self.relative.reset()
        self.relative_trace = None
        self.dwell.reset()
        self.wink_detector.reset()
        self.wink_sequence.reset()
        self.wink_position = None
        self.blink_guard.reset()
        self.blink_position = None
        if self.eye_calibration is not None:
            self.eye_calibration = self.eye_window = None
            self.eye_button.setEnabled(True)
            self.eye_button.setText("Calibrar ojos")
            self.left_eye_button.setEnabled(self.wink_thresholds is not None)
            self.eye_hint.setText("Calibración ocular cancelada.")
        if self.recentering is not None:
            self.recentering = None
            self.relative_status.setText("Recentrado detenido. Se conserva el neutral anterior.")
            self.arm.setEnabled(True)
        self.arm.setText("Activar cursor")
        self.motion.setText("Cursor pausado")
        self.control_note = "Actuador pausado"
        self.refresh_wink_status()

    def toggle(self):
        if self.armed:
            self.pause()
        else:
            if (self.model_switch_pending or self.recentering is not None or self.eye_calibration is not None
                    or self.stop_pressed()):
                self.log_eye_event("intento de armado", "Bloqueado por pausa de emergencia, calibración o cambio de modelo.")
                return
            with self.vision.lock:
                sample = self.vision.latest
                self.armed_losses = self.vision.losses
            if sample is not None and sample[1] is not None and monotonic() - sample[0] <= .25:
                if self.control.currentIndex() == 1 and self.current_neutral() is None:
                    self.relative_status.setText("El neutral RAW inicial no es estable. Pulsa Recentrar neutral relativo.")
                    return
                self.armed = True
                self.wink_detector.reset()
                self.wink_sequence.reset()
                if self.face_rearm and self.control.currentIndex() == 0:
                    self.safety.soften_after_face_loss()
                    self.face_rearm = False
                self.arm.setText("Pausar cursor")
                self.refresh_wink_status()
            else:
                self.status.setText("Cursor pausado. Inicia la cámara, mide el neutral y espera detección válida.")
                self.log_eye_event("intento de armado", "Bloqueado: falta una lectura facial válida y reciente.")

    def stop_pressed(self):
        return (self.api.GetAsyncKeyState(0x77) | self.api.GetAsyncKeyState(0x1B)) & 0x8000

    def tick(self):
        f8 = bool(self.api.GetAsyncKeyState(0x77) & 0x8000)
        if (f8 and not self.f8) or self.api.GetAsyncKeyState(0x1B) & 0x8000:
            self.pause()
        self.f8 = f8
        self.refresh_wink_status()
        if self.model_switch_pending:
            return
        with self.vision.lock:
            sample = self.vision.latest
            losses = self.vision.losses
            eyes = self.vision.eyes
            pose = self.vision.pose
        if self.vision.isRunning():
            self.mode.model().item(1).setEnabled(self.vision.matrix is not None)
            self.mode.model().item(2).setEnabled(self.vision.reach is not None)
            self.mode.model().item(3).setEnabled(self.vision.coverage is not None)
            self.reset_reach_button.setEnabled(self.vision.reach is not None)
            self.reach_status.setText(self.vision.reach_status)
            self.coverage_status.setText(self.vision.coverage_status)
        now = monotonic()
        if self.recentering is not None:
            self.update_recenter(sample, losses, now)
        blinking = self.protect_blink(sample, eyes, losses, now)
        wink_holding = self.update_winks(sample, eyes, losses, now)
        if not blinking and not wink_holding:
            self.advance(sample, losses, now, pose)
        self.update_accessibility(sample, eyes, losses, now)
        self.refresh_demo_state()
        if self.diagnostic_toggle.isChecked():
            self.show_diagnostics(sample, now)

    def advance(self, sample, losses, now, pose=None):
        if not self.armed:
            return
        if sample is None or sample[1] is None or now - sample[0] > .25 or losses != self.armed_losses:
            self.pause()
            self.face_rearm = True
            self.motion.setText("Pausado por pérdida de detección")
            self.control_note = "Pausa por pérdida de detección o muestra caducada"
            return
        stamp, raw = sample
        if stamp == self.last_sample:
            return
        self.last_sample = stamp
        if self.control.currentIndex() == 1:
            self.advance_relative(raw, stamp, now)
            return
        mode = self.mode.currentIndex()
        if mode == 1 and self.vision.matrix is not None:
            point, mode_name = affine(raw, self.vision.matrix), "Afín"
        elif mode == 2 and self.vision.reach is not None:
            point, mode_name = self.vision.reach.apply(raw), "Alcance"
        elif mode == 3 and self.vision.coverage is not None:
            point, mode_name = self.vision.coverage.apply(raw), "Cobertura 2D"
        else:
            point, mode_name = raw, "RAW"
        current = np.asarray(self.sink.position())
        if self.fixation_active() and np.all((0 <= current) & (current <= self.scale)):
            released = self.fixation.step(stamp, raw, pose, current, self.scale, self.safety.last_sent is not None)
            if released:
                self.safety.resume_after_reopen(now)
                if isinstance(self.fixation, ConsensusFixation):
                    point = self.fixation.release_target
                    if isinstance(self.safety.smooth, ConsensusSmoother):
                        self.safety.smooth.fine_direction = self.fixation.fine_direction
        elif self.fixation.samples or self.fixation.anchor is not None:
            if self.fixation.anchor is not None:
                self.safety.resume_after_reopen(now)
            self.fixation.reset()
            if isinstance(self.safety.smooth, ConsensusSmoother):
                self.safety.smooth.fine_direction = None
        self.update_fixation_label()
        if self.fixation.anchor is not None:
            if self.send(self.fixation.anchor):
                self.motion.setText("Cursor activo · RAW · anclado")
            return
        output = self.safety.step(point, current, now, True, self.scale)
        if output is None:
            self.dwell.suspend()
            self.control_note = ("Puntero fuera del monitor principal" if np.any((current < 0) | (current > self.scale))
                                 else "Esperando una detección estable para reanudar")
            return
        if not self.send(output):
            return
        self.last_command = (now, self.safety.trace, output, raw.copy(), mode_name, point)
        self.motion.setText(f"Cursor activo · {mode_name}")

    def advance_relative(self, raw, stamp, now):
        current = np.asarray(self.sink.position())
        step = self.relative.step(raw, self.current_neutral(), current, stamp, self.scale)
        self.relative_trace = step
        if step is None:
            self.pause()
            self.control_note = "Relativo pausado por neutral no disponible o puntero fuera del monitor principal"
            return
        self.motion.setText(f"Cursor activo · Relativo · {step.state}")
        if np.array_equal(step.output, current):
            self.control_note = "Sin cambio de píxel en modo relativo"
        else:
            with self.vision.lock:
                lost = self.vision.losses != self.armed_losses
            if lost:
                self.pause()
                self.control_note = "Pausa por pérdida de detección"
                return
            if not self.send(step.output):
                return

    def send(self, output):
        if self.stop_pressed():
            self.pause()
            return False
        if not self.api.SetCursorPos(int(output[0]), int(output[1])):
            self.pause()
            self.control_note = f"Windows rechazó el envío de {xy(output)}"
            return False
        self.control_note = "Envío aceptado por Windows"
        return True

    def show_diagnostics(self, sample, now):
        raw = sample[1] if sample is not None and now - sample[0] <= .25 else None
        cursor = np.asarray(self.sink.position())
        live_size = np.array([self.api.GetSystemMetrics(0), self.api.GetSystemMetrics(1)])
        dpi = self.api.GetDpiForSystem()
        dpi_label = f"{100 * dpi / 96:.0f}%" if dpi else "no disponible"
        rect = wintypes.RECT()
        has_rect = bool(self.api.GetClipCursor(ctypes.byref(rect)))
        if has_rect:
            win_rect = f"X [{rect.left}, {rect.right}), Y [{rect.top}, {rect.bottom})"
        else:
            win_rect = "no disponible"
        w, h = self.scale + 1
        lines = [
            f"Modelo seleccionado: {self.model.currentText()}",
            f"Monitor principal Win32: {live_size[0]} × {live_size[1]} · DPI {dpi_label} · Qt DPR {self.devicePixelRatioF():g}",
            f"Conversión usada: {w} × {h} · límites X [0, {w-1}], Y [0, {h-1}]",
            f"Rectángulo permitido por Windows: {win_rect}",
            f"RAW actual [normalizado]: {xy(raw, 5)} · fuera de [0, 1]: "
            + (axes((raw < 0) | (raw > 1)) if raw is not None else "—"),
            f"Cursor real: {xy(cursor)} px · {screen_sides(cursor, self.scale)}",
            self.control_note,
        ]
        if not np.array_equal(live_size, self.scale + 1):
            lines.append("La resolución actual difiere de la usada al abrir la aplicación.")
        if self.control.currentIndex() == 1:
            lines.append(f"Neutral RAW relativo: {xy(self.current_neutral(), 5)} · zona muerta {self.relative.deadzone:g}"
                         f" · máximo {self.relative.speed:g} px/s")
            if self.relative_trace is not None:
                step = self.relative_trace
                lines += [f"Relativo · {step.state} · desviación RAW {xy(step.deviation, 5)}",
                          f"Velocidad suavizada: {xy(step.velocity, 1)} px/s · clipping: {axes(step.clipped)}",
                          f"Destino de este paso: {xy(step.output)} px · cursor real menos destino {xy(cursor-step.output)} px"]
        elif self.last_command is not None:
            sent_at, trace, sent, sent_raw, mode, mapped = self.last_command
            lines += [
                f"Último envío · {mode} · hace {max(0, now-sent_at):.2f} s",
                f"RAW de ese envío: {xy(sent_raw, 5)}",
                f"Tras aplicar el modo [normalizado]: {xy(mapped, 5)}",
                f"Tras suavizado [normalizado]: {xy(trace.smoothed, 5)} · clipping: {axes(trace.clipped)}",
                f"Destino tras clipping: {xy(trace.target_px, 1)} px",
                f"Limitador: {'activo' if trace.limited else 'inactivo'} · distancia {trace.distance_px:.1f} px"
                f" · máximo {trace.limit_px:.1f} px/paso",
                f"Enviado a Windows: {xy(sent)} px · cursor real menos envío: {xy(cursor-sent)} px",
            ]
            if has_rect and not (rect.left <= sent[0] < rect.right and rect.top <= sent[1] < rect.bottom):
                lines.append("El último envío cae fuera del rectángulo permitido por Windows.")
        else:
            lines += ["Tras suavizado: — · clipping: — · limitador: —", "Enviado a Windows: —"]
        lines += self.reach_diagnostics(raw)
        text = "\n".join(lines)
        if self.diagnostic.toPlainText() != text:
            scroll = self.diagnostic.verticalScrollBar()
            position = scroll.value()
            self.diagnostic.setPlainText(text)
            scroll.setValue(position)

    def reach_diagnostics(self, raw):
        mapping = self.vision.reach
        if mapping is None:
            return ["ALCANCE · " + self.vision.reach_status]
        x, y = mapping.knots
        gx, gy = mapping.gains
        lines = [
            f"ALCANCE · Perfil cargado y compatible: {mapping.source or '—'}",
            f"Anclas RAW X (izq./neutral/der.): {x[0]:.5f} / {x[1]:.5f} / {x[2]:.5f}",
            f"Anclas RAW Y (arriba/neutral/abajo): {y[0]:.5f} / {y[1]:.5f} / {y[2]:.5f}",
            f"Ganancias X {gx[0]:.3f}× / {gx[1]:.3f}× · Y {gy[0]:.3f}× / {gy[1]:.3f}× · máximo {MAX_GAIN:g}×",
        ]
        if raw is not None:
            mapped = mapping.apply(raw)
            sides = ", ".join(name for name, reached in zip(("izquierda", "derecha", "arriba", "abajo"),
                            mapping.reached_sides(raw)) if reached) or "ninguno"
            lines += [f"Entrada actual ALCANCE: {xy(raw, 5)}",
                      f"Salida ALCANCE: {xy(mapped, 5)} · {xy(mapped * self.scale, 1)} px antes del suavizado",
                      f"Bordes que pide esa salida: {sides}"]
        if self.vision.reach_neutral is None:
            lines.append("Neutral de la sesión: pendiente")
        else:
            current, current_raw = self.vision.reach_neutral
            lines += [
                f"Neutral RAW del perfil: {xy(mapping.knots[:, 1], 5)}",
                f"Neutral RAW de la sesión: {xy(current_raw, 5)} · diferencia {xy(current_raw-mapping.knots[:, 1], 5)}",
                f"Neutral de la sesión en ALCANCE: {xy(mapping.apply(current_raw)*self.scale, 1)} px",
            ]
            if mapping.neutral_reference is not None:
                angle, center, span = compare_neutral(mapping.neutral_reference, current)
                lines.append(f"Referencia facial, sesión menos perfil: yaw {angle[0]:+.2f}°, pitch {angle[1]:+.2f}°, "
                             f"roll {angle[2]:+.2f}° · centro {xy(center, 1)} px de cámara · escala {span:+.1f}%")
        return lines

    def closeEvent(self, event):
        self.keyboard_dock.cancel()
        self.pause()
        self.vision.requestInterruption()
        if not self.vision.wait(100):
            event.ignore()
            QTimer.singleShot(150, self.close)
            return
        if self.actions.dragging:
            event.ignore()
            QTimer.singleShot(150, self.close)
            return
        self.access_bar.close()
        self.dwell_ring.close()
        super().closeEvent(event)


def main(consensus=True, guided_v2=True):
    app = QApplication(sys.argv[:1])
    window = Window(consensus=consensus, presentation=True)
    if guided_v2:
        window.model.setCurrentIndex(2)
    window.show()
    return app.exec()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HeadControl AI para Windows")
    parser.parse_args()
    raise SystemExit(main())
