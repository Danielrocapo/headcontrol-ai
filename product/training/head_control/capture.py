"""Interfaz de captura de muestras reales para HeadControlNet."""

import argparse
from collections import Counter
from contextlib import ExitStack
import math
from pathlib import Path
import re
import sys
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

from features import CameraFeatures, RAW, VERSION, relative
from storage import (
    CLASSES, MAX_SAMPLES, MEASURE_SECONDS, MIN_SAMPLES, ROW_FIELDS, SAMPLE_INTERVAL,
    SETTLE_SECONDS, Store, dumps, make_plan, write_csv,
)

HERE = Path(__file__).resolve().parent
WINDOW = "HeadControlNet | Captura de movimientos"
INSTRUCTIONS = {
    "CENTRO": "Mira al frente con la cabeza recta.",
    "IZQUIERDA": "Gira la cabeza hacia tu izquierda.",
    "DERECHA": "Gira la cabeza hacia tu derecha.",
    "ARRIBA": "Levanta ligeramente el menton.",
    "ABAJO": "Baja ligeramente el menton.",
}


class StopCapture(Exception):
    pass


class Interface:
    def __init__(self, camera):
        self.camera = camera
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 820, 670)

    def frame(self, title, instructions, show_preview=True):
        now, row, rgb, landmarks = self.camera.read()
        canvas = np.full((670, 820, 3), (22, 24, 28), dtype=np.uint8)
        if rgb is not None and show_preview:
            preview = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (800, 450))
            for index in (1, 33, 152, 263):
                if index < len(landmarks):
                    point = landmarks[index]
                    if math.isfinite(point.x) and math.isfinite(point.y):
                        cv2.circle(preview, (round(point.x * 800), round(point.y * 450)), 4, (0, 225, 255), -1)
            canvas[10:460, 10:810] = preview
        valid = row["valid_detection"]
        angles = " | ".join(f"{name} {row[field]:+.1f}" for name, field in
                            zip(("Yaw", "Pitch", "Roll"), RAW[:3])) if valid else row["status"]
        cv2.putText(canvas, title, (18, 500), cv2.FONT_HERSHEY_SIMPLEX, .85, (186, 229, 255), 2, cv2.LINE_AA)
        for i, line in enumerate((*instructions, ("Valida | " if valid else "Sin muestra | ") + angles)):
            cv2.putText(canvas, line, (18, 538 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, .59,
                        (230, 232, 235), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xff
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise StopCapture
        return now, row, key

    def wait(self, title, lines, keys):
        while True:
            _, _, key = self.frame(title, lines)
            if key in keys:
                return key


def calibrate(ui, store):
    instructions = ("Mira al frente, con el cuello relajado.", "ESPACIO para calibrar | ESC para salir.")
    while True:
        ui.wait("Referencia neutral", instructions, (32,))
        start, values = monotonic(), []
        while monotonic() - start < 2.5:
            now, row, _ = ui.frame("Midiendo referencia neutral", ("Manten la postura comoda.",))
            if now - start >= .5 and row["valid_detection"]:
                values.append([row[k] for k in RAW])
        if len(values) >= 12:
            spread = np.ptp(np.percentile(np.asarray(values)[:, :3], (10, 90), axis=0), axis=0)
            if np.all(spread <= 5):
                middle = np.median(values, axis=0)
                key = store.baseline(middle, ui.camera.metadata,
                                     dict(valid_frames=len(values), p90_minus_p10_deg=spread.tolist()))
                return key, middle
        instructions = ("Referencia inestable o menos de 12 fotogramas validos.",
                        "Vuelve al centro. ESPACIO para repetir | ESC para salir.")


def record(ui, store, trial, baseline_id, baseline, progress):
    title = f'{progress} | {trial["label"]}'
    ui.wait(title, (INSTRUCTIONS[trial["label"]], "Vuelve al centro antes de cada repeticion.",
                    "ESPACIO para empezar | ESC para salir."), (32,))
    rows, quality = [], Counter()
    start, last_sample = monotonic(), -math.inf
    try:
        while True:
            elapsed = monotonic() - start
            measuring = elapsed >= SETTLE_SECONDS
            now, row, key = ui.frame(title, (
                INSTRUCTIONS[trial["label"]],
                "MIDIENDO: manten la postura." if measuring else "PREPARA EL MOVIMIENTO.",
                "R para repetir | ESC para salir.",
            ))
            if key in (ord("r"), ord("R")):
                store.attempt(trial, baseline_id, "repeated", rows, quality)
                return
            if measuring and now - start < SETTLE_SECONDS + MEASURE_SECONDS:
                quality[row["status"]] += 1
                if (row["valid_detection"] and len(rows) < MAX_SAMPLES
                        and now - last_sample >= SAMPLE_INTERVAL):
                    row.update(relative([row[k] for k in RAW], baseline))
                    rows.append(row)
                    last_sample = now
            if now - start >= SETTLE_SECONDS + MEASURE_SECONDS:
                break
        good = len(rows) >= MIN_SAMPLES
        key = ui.wait("Medicion terminada", (
            f"{len(rows)} muestras validas. Descansa y vuelve al centro.",
            "ENTER si hiciste el movimiento indicado | R para repetir." if good
            else f"Se necesitan {MIN_SAMPLES} muestras validas. R para repetir.",
            "ESC para salir.",
        ), (10, 13, ord("r"), ord("R")) if good else (ord("r"), ord("R")))
    except StopCapture:
        store.attempt(trial, baseline_id, "cancelled", rows, quality)
        raise
    status = "accepted" if key in (10, 13) else ("repeated" if good else "insufficient_detection")
    store.attempt(trial, baseline_id, status, rows, quality)


def capture(store):
    if not store.pending():
        return
    with ExitStack() as stack:
        camera = CameraFeatures(stack)
        ui = Interface(camera)
        stack.callback(cv2.destroyAllWindows)
        try:
            baseline_id, baseline = calibrate(ui, store)
            while pending := store.pending():
                number = len(store.config["plan"]) - len(pending) + 1
                record(ui, store, pending[0], baseline_id, baseline, f'{number}/{len(store.config["plan"])}')
            ui.wait("Sesion completada", ("Las repeticiones ya estan guardadas.", "ESC para salir."), ())
        except StopCapture:
            pass


def smoke():
    folder = HERE / "technical" / uuid4().hex[:12]
    folder.mkdir(parents=True)
    with ExitStack() as stack:
        camera = CameraFeatures(stack)
        rows, start = [], monotonic()
        while monotonic() - start < 10:
            _, row, _, _ = camera.read()
            rows.append(row)
    valid = [row for row in rows if row["valid_detection"]]
    baseline = np.median([[r[k] for k in RAW] for r in valid], axis=0) if valid else None
    for row in valid:
        row.update(relative([row[k] for k in RAW], baseline))
    write_csv(folder / "fotogramas_tecnicos.csv", ROW_FIELDS, rows)
    metadata = dict(camera.metadata, split="technical", status_counts=dict(Counter(r["status"] for r in rows)),
                    baseline_from_test=baseline.tolist() if baseline is not None else None)
    (folder / "metadata.json").write_text(dumps(metadata) + "\n", encoding="utf-8")
    print(f"{len(rows)} fotogramas, {len(valid)} válidos. {folder}")
    return 0 if len(valid) >= MIN_SAMPLES else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--session", help="Identificador único de sesión, también para reanudar")
    mode.add_argument("--export", help="Generar una exportación nueva sin abrir la cámara")
    mode.add_argument("--smoke", action="store_true", help="Prueba técnica de cámara de 10 s, sin etiquetas")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--distance-cm", type=float)
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    name = args.session or args.export
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", name) or name in {
        "con", "aux", "nul", "prn", *(f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10))
    }:
        parser.error("Usa letras minúsculas, números, guion o guion bajo en el identificador.")
    folder = HERE / "sessions" / name
    config = None
    if args.session and not (folder / "session.sqlite3").is_file():
        if (args.split is None or args.distance_cm is None or
                not math.isfinite(args.distance_cm) or args.distance_cm <= 0):
            parser.error("Una sesión nueva requiere --split y --distance-cm positivo.")
        folder.mkdir(parents=True, exist_ok=True)
        config = dict(session_id=name, split=args.split, distance_cm=args.distance_cm, contract=VERSION,
                      classes=CLASSES, plan=make_plan(name), settle_s=SETTLE_SECONDS,
                      measurement_s=MEASURE_SECONDS, sample_interval_s=SAMPLE_INTERVAL,
                      min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES)
    with Store(folder, config) as store:
        if store.config["contract"] != VERSION:
            raise RuntimeError("El contrato del capturador no coincide con la sesión.")
        if any(v is not None and v != store.config[k] for k, v in
               (("split", args.split), ("distance_cm", args.distance_cm))):
            parser.error("La partición y distancia declaradas son fijas por sesión.")
        try:
            if args.session:
                capture(store)
        finally:
            print(f"Exportación: {store.export()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, cv2.error) as exc:
        sys.exit(str(exc))
