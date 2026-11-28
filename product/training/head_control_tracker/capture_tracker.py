"""Captura supervisada de coordenadas para HeadControlNet Tracker."""

import argparse
from collections import Counter
from contextlib import ExitStack
import ctypes
import math
from pathlib import Path
import re
import sys
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "head_control"))

from capture import StopCapture, calibrate
from features import CameraFeatures, FEATURES, RAW, VERSION, relative
from storage import MAX_SAMPLES, MEASURE_SECONDS, MIN_SAMPLES, SAMPLE_INTERVAL, SETTLE_SECONDS
from session import CONTRACT, TrackerStore, make_plan

WINDOW = "HeadControlNet Tracker"
YELLOW = (0, 229, 255)
REPEAT = (ord("r"), ord("R"))
CONFIRM = (10, 13)


def primary_size():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class Screen:
    def __init__(self, camera, width, height):
        self.camera = camera
        self.width, self.height = width, height
        self.buffer = np.empty((height, width, 3), dtype=np.uint8)
        self.radius = max(4, round(min(width, height) * .012))
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.moveWindow(WINDOW, 0, 0)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def frame(self, title, instructions, trial=None, mark=True):
        now, row, _, _ = self.camera.read()
        self.buffer[:] = (23, 26, 30)
        scale = max(.45, min(1.1, self.width / 1280))
        x_text = round(self.width * .04)
        y_text = round(self.height * .23)
        step = round(29 * scale)
        status = "Rostro y pose validos" if row["valid_detection"] else f"Sin muestra: {row['status']}"
        for i, line in enumerate((title, *instructions, status)):
            color = (240, 242, 245) if i == 0 else (192, 199, 209)
            cv2.putText(self.buffer, line, (x_text, y_text + i * step), cv2.FONT_HERSHEY_SIMPLEX,
                        .67 * scale, color, 2 if i == 0 else 1, cv2.LINE_AA)
        if mark:
            at = ((round((self.width - 1) / 2), round((self.height - 1) / 2)) if trial is None
                  else (trial["target_px_x"], trial["target_px_y"]))
            cv2.circle(self.buffer, at, self.radius, YELLOW, -1, cv2.LINE_AA)
        return self.present(now, row)

    def present(self, now, row):
        cv2.imshow(WINDOW, self.buffer)
        key = cv2.waitKey(1) & 0xff
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise StopCapture
        bounds = cv2.getWindowImageRect(WINDOW)
        if bounds != (0, 0, self.width, self.height):
            raise RuntimeError(f"Área de dibujo {bounds}; se requiere (0, 0, {self.width}, {self.height}).")
        return now, row, key

    def wait(self, title, lines, keys, trial=None, mark=True):
        while True:
            _, _, key = self.frame(title, lines, trial, mark)
            if key in keys:
                return key


def record(view, store, trial, baseline_id, baseline, technical):
    title = f'{trial["target_id"]}  |  REPETICION {trial["round"]}'
    if not technical:
        view.wait(title, ("Vuelve al centro. Mueve la cabeza hacia el punto amarillo.",
                          "ESPACIO para iniciar | ESC para guardar y salir."), (32,), trial)
    quality, rows = Counter(), []
    start, previous = monotonic(), -math.inf
    try:
        while True:
            measuring = monotonic() - start >= SETTLE_SECONDS
            phase = "MIDIENDO | Manten la cabeza en el objetivo." if measuring else "ESTABILIZANDO"
            mode = "PRUEBA TECNICA | ESC para salir." if technical else "R para repetir | ESC para salir."
            now, row, key = view.frame(title, (phase, mode), trial)
            if key in REPEAT and not technical:
                store.attempt(trial, baseline_id, "repeated", rows, quality)
                return False
            if now - start >= SETTLE_SECONDS + MEASURE_SECONDS:
                break
            if measuring:
                quality[row["status"]] += 1
                if row["valid_detection"] and now - previous >= SAMPLE_INTERVAL and len(rows) < MAX_SAMPLES:
                    row.update(relative([row[k] for k in RAW], baseline))
                    rows.append(row)
                    previous = now
        good = len(rows) >= MIN_SAMPLES
        if technical:
            status = "accepted" if good else "insufficient_detection"
        else:
            key = view.wait("MEDICION TERMINADA", (
                f"{len(rows)} muestras validas.",
                "ENTER si orientaste la cabeza al punto | R para repetir." if good
                else f"Se necesitan {MIN_SAMPLES} muestras validas. Pulsa R.", "ESC para guardar y salir.",
            ), CONFIRM + REPEAT if good else REPEAT, trial)
            status = "accepted" if key in CONFIRM else ("repeated" if good else "insufficient_detection")
    except StopCapture:
        store.attempt(trial, baseline_id, "cancelled", rows, quality)
        raise
    store.attempt(trial, baseline_id, status, rows, quality)
    return status == "accepted"


def technical_reference(view, store):
    values = []
    start = monotonic()
    while monotonic() - start < 2.5:
        _, row, _ = view.frame("PRUEBA TECNICA", ("Referencia de camara. Sin captura de entrenamiento.",))
        if row["valid_detection"]:
            values.append([row[k] for k in RAW])
    if len(values) < 12:
        raise RuntimeError(f"Se requieren 12 detecciones válidas para la referencia técnica; se obtuvieron {len(values)}.")
    baseline = np.median(values, axis=0)
    return store.baseline(baseline, view.camera.metadata, dict(valid_frames=len(values))), baseline


def capture(store):
    if not store.pending():
        return True
    technical = store.config["split"] == "technical"
    width, height = store.config["screen_size"]
    if (width, height) != primary_size():
        raise RuntimeError("La resolución del monitor principal difiere de la registrada en esta sesión.")
    with ExitStack() as stack:
        camera = CameraFeatures(stack)
        stack.callback(cv2.destroyAllWindows)
        view = Screen(camera, width, height)
        try:
            ref, baseline = (technical_reference(view, store) if technical else calibrate(view, store))
            if technical:
                for trial in store.pending():
                    record(view, store, trial, ref, baseline, True)
            else:
                while pending := store.pending():
                    record(view, store, pending[0], ref, baseline, False)
                view.wait("SESION COMPLETA", ("Capturas guardadas | ESC para salir.",), (), mark=False)
        except StopCapture:
            pass
    return not store.pending()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--session", help="Crear o reanudar una sesión")
    mode.add_argument("--export", help="Exportar una sesión sin abrir la cámara")
    mode.add_argument("--smoke", action="store_true", help="Comprobar nueve objetivos con cámara real")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--distance-cm", type=float)
    parser.add_argument("--grid", choices=("3x3", "5x3"))
    parser.add_argument("--rounds", type=int, choices=range(2, 13), metavar="2-12")
    args = parser.parse_args()
    name = f"technical-{uuid4().hex[:12]}" if args.smoke else (args.session or args.export)
    if not re.fullmatch(r"tr-[a-z0-9][a-z0-9_-]{0,44}", name) and not args.smoke:
        parser.error("Usa un identificador como tr-01, con minúsculas, cifras y guiones.")
    folder = HERE / ("technical" if args.smoke else "sessions") / name
    config = None
    if args.smoke:
        if any(v is not None for v in (args.split, args.grid, args.rounds)):
            parser.error("La prueba técnica usa una sola ronda de 3x3 con partición technical.")
    if args.smoke or (args.session and not (folder / "session.sqlite3").is_file()):
        if not args.smoke and (args.split is None or args.distance_cm is None):
            parser.error("Una sesión nueva requiere --split y --distance-cm.")
        if args.distance_cm is not None and (not math.isfinite(args.distance_cm) or args.distance_cm <= 0):
            parser.error("La distancia debe ser finita y positiva.")
        width, height = primary_size()
        grid = args.grid or "3x3"
        rounds = 1 if args.smoke else (args.rounds or 4)
        config = dict(session_id=name, split="technical" if args.smoke else args.split, distance_cm=args.distance_cm,
                      grid=grid, rounds=rounds, screen_size=[width, height], contract=CONTRACT,
                      feature_contract=VERSION, features=FEATURES, settle_s=SETTLE_SECONDS,
                      measurement_s=MEASURE_SECONDS, sample_interval_s=SAMPLE_INTERVAL,
                      min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES,
                      plan=make_plan(name, int(grid[0]), rounds, width, height))
        folder.mkdir(parents=True, exist_ok=True)
    with TrackerStore(folder, config) as store:
        if store.config["contract"] != CONTRACT or store.config["feature_contract"] != VERSION:
            raise RuntimeError("Contrato incompatible con la sesión.")
        for field in ("split", "grid", "rounds", "distance_cm"):
            value = getattr(args, field)
            if value is not None and value != store.config[field]:
                parser.error(f"{field} es fijo por sesión.")
        complete = False
        try:
            if args.session or args.smoke:
                complete = capture(store)
        finally:
            print(f"Exportación: {store.export()}")
    return int(args.smoke and not complete)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, cv2.error) as exc:
        sys.exit(str(exc))
