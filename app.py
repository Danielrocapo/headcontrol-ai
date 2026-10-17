"""HeadControl AI: ejecutar con python app.py."""
import json
import logging
import math
import secrets
import sys
import threading
import time
from pathlib import Path

import pyautogui
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import make_server
from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtWebEngineCore import (
    QWebEnginePage, QWebEnginePermission, QWebEngineProfile,
    QWebEngineUrlRequestInterceptor,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

BASE = Path(__file__).resolve().parent
CLASSES = {"CENTRO", "IZQUIERDA", "DERECHA", "ARRIBA", "ABAJO"}
THRESHOLD = 0.85
STABLE_FRAMES = 3
VELOCIDAD = 120  # pixeles por segundo; prueba 60 para ir mas lento o 180 para ir mas rapido
INTERVALO_MOVIMIENTO = 0.02  # segundos entre micropasos (20 ms)
ACELERACION = 360  # pixeles/segundo al cuadrado: arranque gradual
ZONA_MUERTA = 0.12  # segundos de direccion estable antes de arrancar
ZONA_MUERTA_CENTRO = 0.12  # detener si CENTRO conserva al menos esta probabilidad
CADUCIDAD_PREDICCION = 0.30  # detener si dejan de llegar predicciones validas
MAX_FRAME_GAP = 0.75
POSE_DIR = BASE / "static/posenet/mobilenet/float/075"
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # el temporizador limita la frecuencia sin bloquear la interfaz


def validate_assets():
    """Validar los manifiestos sin escribir en model/."""
    required = [BASE / "static" / name for name in (
        "index.html", "tf.min.js", "teachablemachine-pose.min.js")]
    required += [BASE / "model" / name for name in (
        "model.json", "metadata.json", "weights.bin")]
    required += [POSE_DIR / "model-stride16.json"]
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Falta un archivo local o esta vacio: {path}")
    metadata = json.loads((BASE / "model/metadata.json").read_text(encoding="utf-8"))
    if len(metadata.get("labels", [])) != 5 or set(metadata["labels"]) != CLASSES:
        raise ValueError("Clases esperadas: CENTRO, IZQUIERDA, DERECHA, ARRIBA, ABAJO.")
    settings = metadata.get("modelSettings", {}).get("posenet", {})
    expected = {"architecture": "MobileNetV1", "outputStride": 16,
                "inputResolution": 257, "multiplier": 0.75}
    if any(settings.get(key, value) != value for key, value in expected.items()):
        raise ValueError("Los pesos PoseNet locales no coinciden con metadata.json.")
    for manifest in (BASE / "model/model.json", POSE_DIR / "model-stride16.json"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for group in data["weightsManifest"]:
            for relative in group["paths"]:
                path = (manifest.parent / relative).resolve()
                if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
                    raise ValueError(f"Pesos locales ausentes o ruta incorrecta: {relative}")


class MouseController:
    def __init__(self):
        self.lock = threading.Lock()
        self.enabled = False
        self.reset()

    def reset(self):
        self.previous = None
        self.count = 0
        self.last_frame = 0.0
        self.since = 0.0
        self.target = None
        self.speed = 0.0
        self.remainder = 0.0
        self.last_tick = time.monotonic()
        self.error = None

    def stop_motion(self):
        """Sin inercia al parar, perder confianza o cambiar de direccion."""
        self.target = None
        self.speed = self.remainder = 0.0

    def reject_prediction(self):
        self.previous, self.count = None, 0
        self.stop_motion()

    def set_enabled(self, enabled):
        with self.lock:
            self.enabled = enabled
            self.reset()

    def prediction(self, direction, confidence, center_confidence=0.0):
        with self.lock:
            if self.error:
                return {"ok": False, "error": self.error}
            if not self.enabled:
                return {"ok": True, "moved": False, "paused": True}
            now = time.monotonic()
            if now - self.last_frame > MAX_FRAME_GAP:
                self.reject_prediction()
            self.last_frame = now
            # CENTRO tiene prioridad inmediata, incluso con poca confianza.
            if direction == "CENTRO":
                self.reject_prediction()
                return {"ok": True, "stable": 0, "state": "center"}
            if confidence < THRESHOLD or center_confidence >= ZONA_MUERTA_CENTRO:
                self.reject_prediction()
                return {"ok": True, "stable": 0, "state": "deadzone"}
            if direction != self.previous:
                self.stop_motion()
                self.since = now
            self.count = min(STABLE_FRAMES, self.count + 1) if direction == self.previous else 1
            self.previous = direction
            ready = self.count >= STABLE_FRAMES and now - self.since >= ZONA_MUERTA
            self.target = direction if ready else None
            return {"ok": True, "stable": self.count, "state": "moving" if ready else "confirming"}

    def update_mouse(self):
        """Micropasos independientes de los FPS del modelo; nunca acumular atrasos."""
        with self.lock:
            now = time.monotonic()
            dt = min(max(0.0, now - self.last_tick), INTERVALO_MOVIMIENTO * 2)
            self.last_tick = now
            if not self.enabled or not self.target:
                return
            if now - self.last_frame > CADUCIDAD_PREDICCION:
                self.reject_prediction()
                return
            self.speed = min(VELOCIDAD, self.speed + ACELERACION * dt)
            distance = self.remainder + self.speed * dt
            pixels = int(distance)
            self.remainder = distance - pixels
            if not pixels:
                return
            dx, dy = {"IZQUIERDA": (-pixels, 0), "DERECHA": (pixels, 0),
                      "ARRIBA": (0, -pixels), "ABAJO": (0, pixels)}[self.target]
            try:
                pyautogui.moveRel(dx, dy, duration=0)
            except pyautogui.FailSafeException:
                self.enabled = False
                self.reset()
                self.error = "FailSafe activado. Aparta el cursor de la esquina y vuelve a iniciar."
            except Exception as exc:
                self.enabled = False
                self.reset()
                logging.exception("No se pudo mover el mouse")
                self.error = f"Error del mouse: {exc}"


def create_app(controller, token):
    app = Flask(__name__, static_folder=str(BASE / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 2048

    @app.before_request
    def local_requests_only():
        if request.remote_addr != "127.0.0.1" or request.host != app.config.get("LOCAL_HOST"):
            return jsonify(ok=False, error="Origen no permitido"), 403
        if request.method == "POST" and not secrets.compare_digest(
                request.headers.get("X-Control-Token", ""), token):
            return jsonify(ok=False, error="Sesion no valida"), 403

    @app.after_request
    def offline_policy(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        html = (BASE / "static/index.html").read_text(encoding="utf-8")
        return (html.replace("__CONTROL_TOKEN__", token)
                .replace("__VELOCIDAD__", str(VELOCIDAD))
                .replace("__INTERVALO__", str(round(INTERVALO_MOVIMIENTO * 1000)))
                .replace("__ZONA_MUERTA__", str(round(ZONA_MUERTA * 1000))))

    @app.get("/model/<path:filename>")
    def model_file(filename):
        return send_from_directory(BASE / "model", filename)

    @app.post("/control")
    def control():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
            return jsonify(ok=False, error="Estado no valido"), 400
        controller.set_enabled(data["enabled"])
        return jsonify(ok=True)

    @app.post("/prediction")
    def prediction():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(ok=False, error="Prediccion no valida"), 400
        direction, confidence = data.get("direction"), data.get("confidence")
        center_confidence = data.get("centerConfidence", 0.0)
        if (not isinstance(direction, str) or direction not in CLASSES
                or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or type(center_confidence) not in (int, float)
                or not math.isfinite(center_confidence) or not 0 <= center_confidence <= 1):
            controller.set_enabled(False)
            return jsonify(ok=False, error="Clase o confianza no valida"), 400
        return jsonify(controller.prediction(direction, confidence, center_confidence))

    return app


class LocalOnly(QWebEngineUrlRequestInterceptor):
    def __init__(self, origin, parent):
        super().__init__(parent)
        self.origin = origin

    def interceptRequest(self, info):
        url = info.requestUrl()
        allowed = (url.scheme() == "http" and url.host() == "127.0.0.1"
                   and url.port() == self.origin.port())
        info.block(not allowed and url.scheme() not in ("data", "blob", "about"))


class Window(QMainWindow):
    def __init__(self, origin, controller):
        super().__init__()
        self.controller = controller
        self.movement_timer = QTimer(self)
        self.movement_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.movement_timer.setInterval(max(1, round(INTERVALO_MOVIMIENTO * 1000)))
        self.movement_timer.timeout.connect(controller.update_mouse)
        self.movement_timer.start()
        self.setWindowTitle("HeadControl AI")
        self.resize(860, 850)
        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)
        self.profile = QWebEngineProfile(self)
        self.interceptor = LocalOnly(origin, self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.page = QWebEnginePage(self.profile, self.view)
        self.view.setPage(self.page)

        def permission_requested(permission):
            url = permission.origin()
            if (url.scheme() == "http" and url.host() == "127.0.0.1"
                    and url.port() == origin.port()
                    and permission.permissionType() == QWebEnginePermission.PermissionType.MediaVideoCapture):
                permission.grant()
            else:
                permission.deny()

        self.page.permissionRequested.connect(permission_requested)
        self.view.loadFinished.connect(self.on_load)
        self.pause_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.pause_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self.pause_shortcut.activated.connect(self.pause)
        self.view.setUrl(origin)

    def on_load(self, success):
        if not success:
            self.controller.set_enabled(False)
            QMessageBox.critical(self, "HeadControl AI", "No se pudo cargar la interfaz local.")

    def pause(self):
        self.controller.set_enabled(False)
        self.page.runJavaScript("window.pauseControl && window.pauseControl()")

    def closeEvent(self, event):
        self.controller.set_enabled(False)
        super().closeEvent(event)


def main():
    qt_app = QApplication(sys.argv)
    try:
        validate_assets()
    except (ValueError, OSError, KeyError) as exc:
        QMessageBox.critical(None, "HeadControl AI - archivos locales", str(exc))
        return 1
    controller = MouseController()
    app = create_app(controller, secrets.token_hex(32))
    # Puerto libre elegido por Windows; escuchar exclusivamente en loopback.
    server = make_server("127.0.0.1", 0, app, threaded=True)
    app.config["LOCAL_HOST"] = f"127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    window = Window(QUrl(f"http://{app.config['LOCAL_HOST']}/"), controller)
    window.show()
    try:
        return qt_app.exec()
    finally:
        controller.set_enabled(False)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        window.view.setPage(None)
        window.page.deleteLater()
        qt_app.processEvents()


if __name__ == "__main__":
    sys.exit(main())
