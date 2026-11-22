"""HeadControl AI v2. Ejecutar: python v2/app.py"""
import ctypes
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget
)
from engine import HeadController, MouthScrollToggle, WinkDetector, LongBlinkDetector, relative_pose


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "assets" / "face_landmarker.task"
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class CameraWorker(QThread):
    frame = Signal(QImage)
    detected = Signal(object)
    problem = Signal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = str(model_path)
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        camera = None
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not camera.isOpened():
                raise RuntimeError("No se pudo abrir la cámara. Comprueba sus permisos.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_timestamp = 0
            failures = 0
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                while self.running:
                    ok, image = camera.read()
                    if not ok:
                        failures += 1
                        if failures > 15:
                            raise RuntimeError("La cámara dejó de responder.")
                        self.msleep(30)
                        continue
                    failures = 0
                    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    timestamp = max(last_timestamp + 1, int(time.monotonic() * 1000))
                    last_timestamp = timestamp
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                    )
                    packet = {"pose": None, "blend": {}, "time": time.monotonic()}
                    if result.face_landmarks:
                        points = result.face_landmarks[0]
                        nose = (points[1].x, points[1].y)
                        eye_left = (points[33].x, points[33].y)
                        eye_right = (points[263].x, points[263].y)
                        packet["pose"] = relative_pose(nose, eye_left, eye_right)
                        if result.face_blendshapes:
                            packet["blend"] = {
                                item.category_name: item.score
                                for item in result.face_blendshapes[0]
                            }
                    self.detected.emit(packet)
                    mirrored = np.ascontiguousarray(rgb[:, ::-1, :])
                    height, width = mirrored.shape[:2]
                    self.frame.emit(
                        QImage(mirrored.data, width, height, width * 3,
                               QImage.Format.Format_RGB888).copy()
                    )
        except Exception as exc:
            self.problem.emit(str(exc))
        finally:
            if camera is not None:
                camera.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeadControl AI · Control sin manos")
        self.resize(920, 760)
        self.head = HeadController()
        self.wink = WinkDetector()
        self.blink = LongBlinkDetector()
        self.mouth = MouthScrollToggle()
        self.enabled = False
        self.last_time = None
        self.last_face = time.monotonic()
        self.last_cursor = None
        self.dwell_start = None
        self.dwell_fired = False
        self.dwell_progress = 0.0
        self.last_action = "Ninguno"
        self.last_action_time = 0.0
        self.key_states = {}
        self.worker = None
        self._build_ui()
        self.key_timer = QTimer(self)
        self.key_timer.timeout.connect(self.poll_hotkeys)
        self.key_timer.start(60)
        self.start_camera()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #101621; color: #eef3fc; font-family: Segoe UI; font-size: 13px; }
            QFrame { background: #182132; border: 1px solid #2d3e57; border-radius: 12px; }
            QPushButton { padding: 11px 14px; background: #293750; border: 1px solid #3e5374;
                          border-radius: 8px; font-weight: 600; }
            QPushButton:hover { background: #365071; }
            QPushButton#primary { background: #52b6ed; color: #081524; }
            QLabel#title { font-size: 27px; font-weight: 700; }
            QLabel#subtitle { color: #b0bfd5; }
            QLabel#preview { border: 1px solid #30415b; border-radius: 10px; background: #080e17; }
            QCheckBox { padding: 5px; }
            QSlider { min-height: 22px; }
        """)
        body = QWidget()
        self.setCentralWidget(body)
        root = QVBoxLayout(body)
        root.setContentsMargins(26, 20, 26, 20)
        root.setSpacing(12)

        title = QLabel("HeadControl AI")
        title.setObjectName("title")
        root.addWidget(title)
        intro = QLabel("Control del cursor con seguimiento facial local · MediaPipe + Python")
        intro.setObjectName("subtitle")
        root.addWidget(intro)

        self.preview = QLabel("Iniciando cámara…")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(360)
        root.addWidget(self.preview, 1)

        self.info = QLabel("Preparando detector facial…")
        self.info.setWordWrap(True)
        root.addWidget(self.info)
        self.gesture_info = QLabel("Gestos: iniciando detector…")
        self.gesture_info.setObjectName("subtitle")
        self.gesture_info.setWordWrap(True)
        root.addWidget(self.gesture_info)

        controls = QHBoxLayout()
        self.toggle_button = QPushButton("Activar control (F8)")
        self.toggle_button.setObjectName("primary")
        self.toggle_button.clicked.connect(self.toggle)
        controls.addWidget(self.toggle_button)
        recenter = QPushButton("Calibrar centro (F9)")
        recenter.clicked.connect(self.calibrate)
        controls.addWidget(recenter)
        self.left_button = QPushButton("Clic izquierdo")
        self.left_button.clicked.connect(lambda: self.manual_click("left"))
        controls.addWidget(self.left_button)
        root.addLayout(controls)

        panel = QFrame()
        panel_layout = QVBoxLayout(panel)
        self.enable_winks = QCheckBox("Guiño de un ojo: clic izquierdo / derecho")
        self.enable_winks.setChecked(True)
        panel_layout.addWidget(self.enable_winks)
        self.enable_blink = QCheckBox("Cerrar ambos ojos 0,5 s: clic izquierdo")
        self.enable_blink.setChecked(True)
        panel_layout.addWidget(self.enable_blink)
        self.enable_mouth = QCheckBox("Abrir la boca 1 segundo: alternar modo scroll")
        self.enable_mouth.setChecked(True)
        panel_layout.addWidget(self.enable_mouth)
        self.enable_dwell = QCheckBox("Clic por permanencia: dejar el cursor quieto 1,5 segundos")
        self.enable_dwell.setChecked(True)
        panel_layout.addWidget(self.enable_dwell)
        self.invert_x = QCheckBox("Invertir movimiento horizontal")
        self.invert_x.setChecked(True)
        self.invert_x.toggled.connect(lambda value: setattr(self.head, "reverse_x", value))
        panel_layout.addWidget(self.invert_x)

        line1 = QHBoxLayout()
        self.speed_label = QLabel("Velocidad: 900 px/s")
        line1.addWidget(self.speed_label)
        speed = QSlider(Qt.Orientation.Horizontal)
        speed.setRange(250, 1500)
        speed.setValue(900)
        speed.valueChanged.connect(self.set_speed)
        line1.addWidget(speed)
        panel_layout.addLayout(line1)

        line2 = QHBoxLayout()
        self.sensitivity_label = QLabel("Sensibilidad: 100 %")
        line2.addWidget(self.sensitivity_label)
        sensitivity = QSlider(Qt.Orientation.Horizontal)
        sensitivity.setRange(50, 220)
        sensitivity.setValue(100)
        sensitivity.valueChanged.connect(self.set_sensitivity)
        line2.addWidget(sensitivity)
        panel_layout.addLayout(line2)

        root.addWidget(panel)
        bottom = QLabel("F8: activar/pausar desde cualquier ventana · F9: recalibrar · "
                        "Esquina de pantalla: parada de emergencia (PyAutoGUI).")
        bottom.setObjectName("subtitle")
        bottom.setWordWrap(True)
        root.addWidget(bottom)

    def start_camera(self):
        if not MODEL.is_file():
            self.info.setText("Falta el modelo facial. Ejecuta: python v2/download_model.py")
            self.toggle_button.setEnabled(False)
            return
        self.worker = CameraWorker(MODEL)
        self.worker.frame.connect(self.show_frame)
        self.worker.detected.connect(self.process)
        self.worker.problem.connect(self.on_problem)
        self.worker.start()

    def show_frame(self, image):
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.width(), self.preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.preview.setPixmap(pixmap)

    def set_speed(self, value):
        self.head.max_speed = float(value)
        self.speed_label.setText(f"Velocidad: {value} px/s")

    def set_sensitivity(self, value):
        self.head.sensitivity = value / 100
        self.sensitivity_label.setText(f"Sensibilidad: {value} %")

    def calibrate(self):
        self.enabled = False
        self.toggle_button.setText("Activar control (F8)")
        self.head.calibrate()
        self.wink.reset()
        self.blink.reset()
        self.mouth.scroll = False
        self.reset_dwell()
        self.info.setText("Mira al frente con comodidad mientras se calibra el centro…")

    def toggle(self):
        if self.worker is None or not self.worker.isRunning():
            return
        if not self.enabled and self.head.neutral is None:
            self.info.setText("Primero mira al frente mientras finaliza la calibración.")
            return
        self.enabled = not self.enabled
        self.wink.reset()
        self.blink.reset()
        self.mouth.scroll = False
        self.reset_dwell()
        self.last_time = None
        self.toggle_button.setText("Pausar control (F8)" if self.enabled else "Activar control (F8)")
        if not self.enabled:
            self.info.setText("Control pausado.")

    def manual_click(self, button):
        if self.enabled:
            pyautogui.click(button=button)

    def on_problem(self, message):
        self.enabled = False
        self.toggle_button.setEnabled(False)
        self.toggle_button.setText("Control detenido")
        self.info.setText("Error: " + message)

    def process(self, packet):
        pose = packet["pose"]
        now = packet["time"]
        if pose is None:
            self.head.observe(None)
            self.wink.reset()
            self.blink.reset()
            self.reset_dwell()
            self.gesture_info.setText("No hay rostro: clics desactivados.")
            if self.enabled and now - self.last_face > 1.0:
                self.enabled = False
                self.toggle_button.setText("Activar control (F8)")
                self.info.setText("No se detecta rostro. Control pausado por seguridad.")
            elif not self.enabled:
                self.info.setText("Colócate frente a la cámara.")
            self.last_time = now
            return

        self.last_face = now
        ready = self.head.observe(pose)
        if not ready:
            self.info.setText(f"Calibrando posición neutral… {round(self.head.progress * 100)} %")
            self.last_time = now
            return
        blend = packet["blend"]
        left_eye = blend.get("eyeBlinkLeft", 0.0)
        right_eye = blend.get("eyeBlinkRight", 0.0)
        jaw = blend.get("jawOpen", 0.0)
        self.show_gestures(left_eye, right_eye, jaw, now)
        if not self.enabled:
            self.info.setText("Rostro detectado. Pulsa F8 para empezar.")
            self.last_time = now
            return
        if self.enable_mouth.isChecked():
            self.mouth.update(blend.get("jawOpen", 0.0), now)
        else:
            self.mouth.scroll = False

        long_blink_clicked = False
        if self.enable_blink.isChecked():
            long_blink_clicked = self.blink.update(left_eye, right_eye, now)
            if long_blink_clicked:
                self.wink.reset()
                self.click_safe("left", "ojos cerrados")
        else:
            self.blink.reset()

        both_closed = (left_eye >= self.blink.threshold and
                       right_eye >= self.blink.threshold)
        if self.enable_winks.isChecked() and not both_closed and not long_blink_clicked:
            gesture = self.wink.update(left_eye, right_eye, now)
            if gesture:
                self.click_safe(gesture, "guiño")
        else:
            self.wink.reset()

        dt = 0.033 if self.last_time is None else min(max(now - self.last_time, 0.001), 0.12)
        self.last_time = now
        dx, dy, scroll = self.head.move(pose, dt, scrolling=self.mouth.scroll)
        try:
            if dx or dy:
                pyautogui.moveRel(dx, dy, duration=0)
            if scroll:
                pyautogui.scroll(scroll)
            self.check_dwell(now)
            self.show_gestures(left_eye, right_eye, jaw, now)
        except pyautogui.FailSafeException:
            self.enabled = False
            self.toggle_button.setText("Activar control (F8)")
            self.info.setText("Parada de emergencia activada. Usa F8 para reanudar.")
            return
        except Exception as exc:
            self.enabled = False
            self.info.setText("Movimiento detenido por error: " + str(exc))
            return

        self.info.setText(
            "Control activo · " +
            ("MODO SCROLL · abre la boca 1 segundo para salir"
             if self.mouth.scroll else "MODO CURSOR · mantén quieto para clic")
        )

    def click_safe(self, button, source="gesto"):
        if not self.enabled:
            return
        try:
            pyautogui.click(button=button)
            kind = "izquierdo" if button == "left" else "derecho"
            self.last_action = f"Clic {kind} por {source}"
            self.last_action_time = time.monotonic()
            self.reset_dwell()
            if source != "permanencia":
                self.dwell_fired = True
        except pyautogui.FailSafeException:
            self.enabled = False
            self.toggle_button.setText("Activar control (F8)")
            self.info.setText("Parada de emergencia activada.")
        except Exception as exc:
            self.enabled = False
            self.toggle_button.setText("Activar control (F8)")
            self.info.setText(f"No se pudo hacer clic: {exc}")

    def reset_dwell(self):
        self.dwell_start = None
        self.last_cursor = None
        self.dwell_fired = False
        self.dwell_progress = 0.0

    def show_gestures(self, left_eye, right_eye, jaw, now):
        progress = round(self.dwell_progress * 100)
        action = (f" · Última acción: {self.last_action}"
                  if now - self.last_action_time < 6.0 else "")
        self.gesture_info.setText(
            f"Ojos: izq. {left_eye:.2f} · der. {right_eye:.2f} · "
            f"Boca {jaw:.2f} · Permanencia {progress} %{action}"
        )

    def check_dwell(self, now):
        if not self.enable_dwell.isChecked() or self.mouth.scroll:
            self.reset_dwell()
            return
        position = pyautogui.position()
        if self.last_cursor is None:
            self.last_cursor, self.dwell_start = position, now
            return
        distance = ((position.x - self.last_cursor.x) ** 2 +
                    (position.y - self.last_cursor.y) ** 2) ** 0.5
        if distance > 28:
            self.last_cursor, self.dwell_start = position, now
            self.dwell_fired = False
            self.dwell_progress = 0.0
        elif not self.dwell_fired:
            self.dwell_progress = min(1.0, (now - self.dwell_start) / 1.5)
            if self.dwell_progress >= 1.0:
                self.click_safe("left", "permanencia")
                self.last_cursor = position
                self.dwell_start = now
                self.dwell_fired = True
                self.dwell_progress = 1.0

    def poll_hotkeys(self):
        if os.name != "nt":
            return
        for code, callback in ((0x77, self.toggle), (0x78, self.calibrate)):
            pressed = bool(ctypes.windll.user32.GetAsyncKeyState(code) & 0x8000)
            if pressed and not self.key_states.get(code, False):
                callback()
            self.key_states[code] = pressed

    def closeEvent(self, event):
        self.enabled = False
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3500)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    if os.name != "nt":
        QMessageBox.information(None, "HeadControl AI", 
            "Esta versión está optimizada para Windows. Las teclas globales F8/F9 requieren Windows.")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
