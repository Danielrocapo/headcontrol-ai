r"""
HeadControl AI - prueba en vivo del modelo de seguimiento ocular.

No mueve el cursor ni modifica la aplicacion principal.
Muestra un objetivo amarillo y la prediccion verde del modelo para
comprobar visualmente si el modelo estima donde mira el usuario.
Cada intento guarda un CSV por fotograma y un informe TXT en prueba_ojos.

Guardar en: prueba_ojos/gaze_live_test.py
Ejecutar desde la raiz del repositorio:
  .\.venv\Scripts\python.exe .\prueba_ojos\gaze_live_test.py

Archivos necesarios:
  prueba_ojos/headcontrol_gaze_model.joblib
  v2/assets/face_landmarker.task
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv
import ctypes
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "headcontrol_gaze_model.joblib"
FACE_TASK_PATH = REPO_DIR / "v2" / "assets" / "face_landmarker.task"

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Los mismos nueve puntos utilizados durante la captura del dataset.
TARGETS = [
    (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
    (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
    (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
]

SETTLE_SECONDS = 0.8
TARGET_SECONDS = 3.0
SMOOTHING_ALPHA = 0.22

EYES = {
    "A": {
        "corners": (33, 133),
        "eyelids": (159, 145),
        "iris": 468,
    },
    "B": {
        "corners": (362, 263),
        "eyelids": (386, 374),
        "iris": 473,
    },
}

EXPECTED_FEATURES = [
    "eye_a_x", "eye_a_y", "eye_a_open",
    "eye_b_x", "eye_b_y", "eye_b_open",
]


def get_screen_size():
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return 1920, 1080


def get_eye_features(landmarks, eye_data):
    corner_1 = landmarks[eye_data["corners"][0]]
    corner_2 = landmarks[eye_data["corners"][1]]
    upper = landmarks[eye_data["eyelids"][0]]
    lower = landmarks[eye_data["eyelids"][1]]
    iris = landmarks[eye_data["iris"]]

    if corner_1.x <= corner_2.x:
        left, right = corner_1, corner_2
    else:
        left, right = corner_2, corner_1

    eye_width = right.x - left.x
    if eye_width < 0.0001:
        return None

    iris_x = (iris.x - left.x) / eye_width

    ratio = (iris.x - left.x) / eye_width
    corner_line_y = left.y + ratio * (right.y - left.y)
    iris_y = (iris.y - corner_line_y) / eye_width

    eye_open = abs(lower.y - upper.y) / eye_width

    if eye_open < 0.045:
        return None

    return iris_x, iris_y, eye_open


def extract_model_features(landmarks):
    eye_a = get_eye_features(landmarks, EYES["A"])
    eye_b = get_eye_features(landmarks, EYES["B"])

    if eye_a is None or eye_b is None:
        return None

    values = [
        eye_a[0], eye_a[1], eye_a[2],
        eye_b[0], eye_b[1], eye_b[2],
    ]

    if not all(np.isfinite(value) for value in values):
        return None

    return np.asarray(values, dtype=np.float64)


def draw_eye_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    for eye_data in EYES.values():
        for index in (
            eye_data["corners"][0],
            eye_data["corners"][1],
            eye_data["iris"],
        ):
            point = landmarks[index]
            color = (0, 255, 0) if index == eye_data["iris"] else (0, 220, 255)
            radius = 4 if index == eye_data["iris"] else 2
            cv2.circle(
                frame,
                (int(point.x * width), int(point.y * height)),
                radius,
                color,
                -1,
            )


def put_text(image, text, position, scale=0.7,
             color=(245, 245, 245), thickness=2):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_camera_preview(canvas, frame):
    preview_width = 320
    preview_height = 180
    preview = cv2.resize(frame, (preview_width, preview_height))
    x = canvas.shape[1] - preview_width - 24
    y = 24
    canvas[y:y + preview_height, x:x + preview_width] = preview
    cv2.rectangle(
        canvas,
        (x - 2, y - 2),
        (x + preview_width + 1, y + preview_height + 1),
        (210, 210, 210),
        2,
    )


def draw_target(canvas, target, screen_width, screen_height):
    x = int(target[0] * screen_width)
    y = int(target[1] * screen_height)
    cv2.circle(canvas, (x, y), 25, (0, 220, 255), 3)
    cv2.circle(canvas, (x, y), 6, (0, 220, 255), -1)


def draw_prediction(canvas, prediction, screen_width, screen_height):
    if prediction is None:
        return

    x = int(np.clip(prediction[0], 0.0, 1.0) * screen_width)
    y = int(np.clip(prediction[1], 0.0, 1.0) * screen_height)

    # Punto verde: posicion estimada por nuestro modelo.
    cv2.circle(canvas, (x, y), 18, (0, 255, 80), 3)
    cv2.circle(canvas, (x, y), 5, (0, 255, 80), -1)


def put_centered_text(canvas, label, center, max_width, scale=0.7,
                      color=(245, 245, 245)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1 if scale < 0.6 or max_width < 160 else 2
    (width, _), _ = cv2.getTextSize(label, font, scale, thickness)
    scale *= min(1.0, (max_width - thickness) / width)
    (width, height), _ = cv2.getTextSize(label, font, scale, thickness)
    put_text(canvas, label, (center[0] - width // 2, center[1] + height // 2),
             scale, color, thickness)


@dataclass
class ErrorStats:
    mean_abs: np.ndarray
    median_abs: np.ndarray
    mean_signed: np.ndarray
    median_signed: np.ndarray


def prediction_errors(prediction, target, screen_size):
    if prediction is None:
        return (None, None, None)
    dx, dy = (prediction - target) * screen_size
    return np.hypot(dx, dy), dx, dy


def summarize_errors(samples):
    if not samples:
        return None

    comparison = []
    for errors in np.array(samples).transpose(1, 0, 2):
        absolute = np.abs(errors)
        comparison.append(ErrorStats(
            absolute.mean(axis=0), np.median(absolute, axis=0),
            errors[:, 1:].mean(axis=0), np.median(errors[:, 1:], axis=0),
        ))
    return comparison


def format_metrics(stats, decimals, compact=False):
    labels = ("T", "X", "Y") if compact else ("Total", "X", "Y")
    absolute = "  ".join(
        f"{label} {mean:.{decimals}f}/{median:.{decimals}f}"
        for label, mean, median in zip(labels, stats.mean_abs, stats.median_abs)
    )
    signed = "  ".join(
        f"{label} {mean:+.{decimals}f}/{median:+.{decimals}f}"
        for label, mean, median in zip(("X", "Y"), stats.mean_signed, stats.median_signed)
    )
    return absolute, signed


class EvaluationLog:
    def __init__(self, model_hash, screen_size):
        started = datetime.now()
        self.base_path = BASE_DIR / f"gaze_diagnostics_{started:%Y%m%d_%H%M%S_%f}"
        self.csv_path = self.base_path.with_suffix(".csv")
        self.model_hash = model_hash
        self.screen_size = screen_size
        self.camera_sizes = set()
        self.samples = [[] for _ in TARGETS]
        self.frame_counts = [Counter() for _ in TARGETS]
        self.file = self.csv_path.open("x", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "timestamp_ms", "timestamp_utc", "point_id", "elapsed_target_s",
            "phase", "target_x", "target_y",
            "raw_prediction_x", "raw_prediction_y",
            "smoothed_prediction_x", "smoothed_prediction_y",
            "raw_error_total_px", "raw_error_x_px", "raw_error_y_px",
            "smoothed_error_total_px", "smoothed_error_x_px", "smoothed_error_y_px",
            "tracking_state", "new_eye_detection", "new_valid_detection", "reused_prediction",
            "since_last_valid_detection_s", "included_in_metrics",
            *EXPECTED_FEATURES, "camera_width", "camera_height",
            "screen_width", "screen_height", "model_sha256",
        ])

    def record(self, timestamp_ms, point_index, elapsed, features, raw_prediction,
               visible_prediction, prediction_age, camera_size):
        target = TARGETS[point_index]
        raw_error = prediction_errors(raw_prediction, target, self.screen_size)
        smoothed_error = prediction_errors(visible_prediction, target, self.screen_size)
        measured = elapsed >= SETTLE_SECONDS
        fresh = raw_prediction is not None
        reused = not fresh and visible_prediction is not None
        tracking_state = "new" if fresh else "reused" if reused else "missing"
        included = measured and fresh

        self.camera_sizes.add(camera_size)
        if measured:
            self.frame_counts[point_index][tracking_state] += 1
        if included:
            self.samples[point_index].append((raw_error, smoothed_error))

        self.writer.writerow([
            timestamp_ms, datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            point_index + 1, elapsed, "measurement" if measured else "settling", *target,
            *(raw_prediction if fresh else (None, None)),
            *(visible_prediction if visible_prediction is not None else (None, None)),
            *raw_error, *smoothed_error, tracking_state, int(features is not None),
            int(fresh), int(reused),
            prediction_age, int(included),
            *(features if features is not None else [None] * len(EXPECTED_FEATURES)),
            *camera_size, *self.screen_size, self.model_hash,
        ])

    def finish(self, completed):
        self.file.close()
        point_results = [(len(samples), summarize_errors(samples)) for samples in self.samples]
        overall = summarize_errors([sample for samples in self.samples for sample in samples])
        report_path = self.base_path.with_suffix(".txt")
        camera_sizes = ", ".join(f"{w} x {h}" for w, h in sorted(self.camera_sizes))
        lines = [
            "HeadControl AI - diagnostico de evaluacion en vivo",
            f"Estado: {'completada' if completed else 'interrumpida'}",
            f"CSV: {self.csv_path.name}",
            f"Modelo: {MODEL_PATH}",
            f"Modelo SHA-256: {self.model_hash}",
            f"Camara (fotogramas recibidos): {camera_sizes or 'sin fotogramas'}",
            f"Monitor: {self.screen_size[0]} x {self.screen_size[1]}",
            f"Fijacion: {SETTLE_SECONDS:.1f} s | Duracion por punto: {TARGET_SECONDS:.1f} s",
            f"Factor de suavizado: {SMOOTHING_ALPHA}",
            "Metricas en px (media/mediana) sobre detecciones nuevas tras la fijacion.",
            "Directa: salida del modelo. Suavizada: salida visible en pantalla.",
            "Con signo: X+ derecha, X- izquierda, Y+ abajo, Y- arriba.",
        ]
        overall_counts = Counter()
        for point_counts in self.frame_counts:
            overall_counts.update(point_counts)

        groups = [("GLOBAL", overall_counts, overall)]
        groups.extend(
            (f"P{index}", point_counts, comparison)
            for index, (point_counts, (_, comparison)) in enumerate(
                zip(self.frame_counts, point_results), start=1
            )
        )
        for label, counts, comparison in groups:
            lines.extend([
                "", label,
                f"Detecciones validas: {counts['new']} | "
                f"Sin deteccion ni prediccion: {counts['missing']} | "
                f"Predicciones reutilizadas: {counts['reused']}",
            ])
            if comparison is None:
                lines.append("Sin datos para calcular metricas.")
                continue
            for name, stats in zip(("Directa", "Suavizada"), comparison):
                absolute, signed = format_metrics(stats, 2)
                lines.append(f"{name} | {absolute} | Con signo: {signed}")

        if overall is not None:
            lines.extend(["", "Resumen con el mismo peso para cada punto (media/mediana de medianas):"])
            for index, name in enumerate(("Directa", "Suavizada")):
                medians = np.array([
                    comparison[index].median_abs
                    for _, comparison in point_results if comparison is not None
                ])
                summary = "  ".join(
                    f"{axis} {mean:.2f}/{median:.2f}"
                    for axis, mean, median in zip(
                        ("Total", "X", "Y"), medians.mean(axis=0), np.median(medians, axis=0),
                    )
                )
                lines.append(f"{name} | {summary}")

        with report_path.open("x", encoding="utf-8") as report:
            report.write("\n".join(lines) + "\n")
        print(f"Fotogramas: {self.csv_path}")
        print(f"Informe: {report_path}")
        return point_results, overall


def draw_results(canvas, point_results, overall):
    height, width = canvas.shape[:2]
    scale = min(width / 1280, height / 720, 1.5)
    text_width = int(width * 0.9)
    yellow = (0, 220, 255)
    green = (0, 255, 80)
    valid_points = sum(count > 0 for count, _ in point_results)
    total_detections = sum(count for count, _ in point_results)
    put_centered_text(canvas, "PRUEBA COMPLETADA", (width // 2, int(height * 0.05)),
                      text_width, 0.9 * scale, green)
    put_centered_text(canvas, f"Puntos: {valid_points}/{len(TARGETS)}  |  "
                      f"Detecciones validas: {total_detections}",
                      (width // 2, int(height * 0.10)), text_width, 0.60 * scale)
    put_centered_text(canvas, "Errores en px (media/mediana)  |  D: directa  S: suavizada",
                      (width // 2, int(height * 0.14)), text_width, 0.56 * scale)

    if overall is not None:
        for index, (name, stats) in enumerate(zip(("Directa", "Suavizada"), overall)):
            absolute, signed = format_metrics(stats, 1)
            top = 0.185 + index * 0.082
            put_centered_text(canvas, f"{name}  |  {absolute}",
                              (width // 2, int(height * top)), text_width, 0.63 * scale)
            put_centered_text(canvas, f"Con signo  |  {signed}",
                              (width // 2, int(height * (top + 0.037))),
                              text_width, 0.56 * scale)
    else:
        put_centered_text(
            canvas, "No se registraron predicciones durante la medicion.",
            (width // 2, int(height * 0.22)), text_width, 0.65 * scale,
            (0, 100, 255),
        )

    cell_width = text_width // 3
    grid_top = int(height * 0.36)
    cell_height = int(height * 0.55) // 3
    grid_left = (width - 3 * cell_width) // 2
    gap = max(2, round(5 * scale))

    for index, (count, comparison) in enumerate(point_results):
        row, column = divmod(index, 3)
        left = grid_left + column * cell_width
        top = grid_top + row * cell_height
        center_x = left + cell_width // 2
        cv2.rectangle(canvas, (left + gap, top + gap),
                      (left + cell_width - gap, top + cell_height - gap),
                      (25, 25, 25), -1)

        cell_text_width = cell_width - 4 * gap
        put_centered_text(canvas, f"P{index + 1}  |  {count} validas",
                          (center_x, top + int(cell_height * 0.16)), cell_text_width,
                          0.62 * scale, yellow)
        if comparison is None:
            put_centered_text(canvas, "Sin datos", (center_x, top + cell_height // 2),
                              cell_text_width, 0.56 * scale)
            continue
        for mode_index, (name, stats) in enumerate(zip(("D", "S"), comparison)):
            absolute, signed = format_metrics(stats, 0, compact=True)
            line_y = 0.35 + mode_index * 0.33
            put_centered_text(canvas, f"{name}  {absolute}",
                              (center_x, top + int(cell_height * line_y)),
                              cell_text_width, 0.52 * scale)
            put_centered_text(canvas, f"Signo  {signed}",
                              (center_x, top + int(cell_height * (line_y + 0.15))),
                              cell_text_width, 0.48 * scale)

    put_centered_text(canvas, "T: total  |  X+ derecha, X- izquierda, Y+ abajo, Y- arriba",
                      (width // 2, int(height * 0.93)), text_width, 0.55 * scale)
    put_centered_text(
        canvas, "R = repetir prueba       Q / ESC = salir",
        (width // 2, int(height * 0.97)), text_width, 0.60 * scale,
        (0, 220, 180),
    )


def main():
    if not MODEL_PATH.is_file():
        print("No encuentro el modelo entrenado:")
        print(MODEL_PATH)
        print("Copia headcontrol_gaze_model.joblib dentro de prueba_ojos.")
        return

    if not FACE_TASK_PATH.is_file():
        print("No encuentro el modelo de deteccion facial de MediaPipe:")
        print(FACE_TASK_PATH)
        print("Comprueba que exista v2/assets/face_landmarker.task.")
        return

    model_bytes = MODEL_PATH.read_bytes()
    model_hash = sha256(model_bytes).hexdigest()
    bundle = joblib.load(BytesIO(model_bytes))
    del model_bytes
    model = bundle.get("model")
    feature_names = bundle.get("feature_names")

    if model is None or feature_names != EXPECTED_FEATURES:
        print("El archivo del modelo no tiene las seis caracteristicas esperadas.")
        print("Vuelve a generar headcontrol_gaze_model.joblib con train_gaze_model.py.")
        return

    screen_width, screen_height = get_screen_size()
    screen_size = (screen_width, screen_height)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("No se pudo abrir la camara. Cierra Zoom, Teams u otra app que la use.")
        return

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camara: {actual_width} x {actual_height}")
    print(f"Pantalla: {screen_width} x {screen_height}")
    print("Modelo cargado:", MODEL_PATH)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(FACE_TASK_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window_name = "HeadControl AI - validacion del seguimiento ocular"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    smoothed_prediction = None
    last_prediction_time = None
    last_timestamp = 0
    state = "intro"
    target_index = 0
    target_started_at = 0.0
    evaluation_log = None
    results = None

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("No se pudo leer la camara.")
                    break

                frame = cv2.flip(frame, 1)

                last_timestamp = max(
                    last_timestamp + 1,
                    int(time.monotonic() * 1000),
                )
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                result = landmarker.detect_for_video(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb,
                    ),
                    last_timestamp,
                )

                raw_prediction = None
                features = None
                status = "No se detecta el rostro"

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    draw_eye_landmarks(frame, landmarks)
                    features = extract_model_features(landmarks)

                    if features is not None:
                        prediction = model.predict(features.reshape(1, -1))[0]
                        if np.all(np.isfinite(prediction)):
                            raw_prediction = prediction
                            clipped_prediction = np.clip(raw_prediction, 0.0, 1.0)
                            if smoothed_prediction is None:
                                smoothed_prediction = clipped_prediction
                            else:
                                smoothed_prediction = (
                                    SMOOTHING_ALPHA * clipped_prediction
                                    + (1.0 - SMOOTHING_ALPHA) * smoothed_prediction
                                )
                            status = "Seguimiento activo"
                        else:
                            status = "Prediccion no valida"
                    else:
                        status = "Abre los ojos y mira la pantalla"

                now = time.monotonic()
                if raw_prediction is not None:
                    last_prediction_time = now
                prediction_age = None if last_prediction_time is None else now - last_prediction_time
                visible_prediction = (
                    smoothed_prediction if prediction_age is not None and prediction_age < 0.6
                    else None
                )

                canvas = np.zeros(
                    (screen_height, screen_width, 3),
                    dtype=np.uint8,
                )

                if state == "intro":
                    put_text(canvas, "HEADCONTROL AI", (60, 95), 1.0)
                    put_text(
                        canvas,
                        "Prueba en vivo del modelo de seguimiento ocular",
                        (60, 150),
                        0.72,
                    )
                    put_text(
                        canvas,
                        "Punto AMARILLO: donde debes mirar.",
                        (60, 255),
                        0.78,
                        (0, 220, 255),
                    )
                    put_text(
                        canvas,
                        "Punto VERDE: donde nuestro modelo estima que estas mirando.",
                        (60, 315),
                        0.72,
                        (0, 255, 80),
                    )
                    put_text(
                        canvas,
                        "Sigue los nueve puntos solo con los ojos y mantén la cabeza quieta.",
                        (60, 395),
                        0.68,
                    )
                    put_text(
                        canvas,
                        "ESPACIO = comenzar       Q / ESC = salir",
                        (60, 500),
                        0.78,
                        (0, 220, 180),
                    )

                elif state == "testing":
                    target = TARGETS[target_index]
                    elapsed = now - target_started_at
                    frame_height, frame_width = frame.shape[:2]
                    evaluation_log.record(
                        last_timestamp, target_index, elapsed, features, raw_prediction,
                        visible_prediction, prediction_age, (frame_width, frame_height),
                    )
                    draw_target(canvas, target, screen_width, screen_height)
                    draw_prediction(
                        canvas,
                        visible_prediction,
                        screen_width,
                        screen_height,
                    )

                    put_text(
                        canvas,
                        f"VALIDACION EN VIVO  |  PUNTO {target_index + 1}/9",
                        (45, 55),
                        0.72,
                    )
                    put_text(
                        canvas,
                        "Mira el circulo amarillo. El verde es la prediccion.",
                        (45, 100),
                        0.62,
                    )

                    if elapsed < SETTLE_SECONDS:
                        put_text(
                            canvas,
                            "Fijando la mirada...",
                            (45, screen_height - 70),
                            0.7,
                            (0, 220, 255),
                        )
                    else:
                        put_text(
                            canvas,
                            f"Tiempo restante: {max(0, TARGET_SECONDS - elapsed):.1f} s",
                            (45, screen_height - 70),
                            0.7,
                            (0, 220, 180),
                        )

                    if elapsed >= TARGET_SECONDS:
                        target_index += 1
                        if target_index >= len(TARGETS):
                            completed_log = evaluation_log
                            evaluation_log = None
                            results = completed_log.finish(completed=True)
                            state = "results"
                        else:
                            evaluation_log.file.flush()
                            target_started_at = time.monotonic()

                elif state == "results":
                    draw_results(canvas, *results)

                if state == "intro":
                    draw_camera_preview(canvas, frame)
                if state != "results":
                    put_text(
                        canvas,
                        status,
                        (25, screen_height - 25),
                        0.55,
                        (0, 255, 80) if status == "Seguimiento activo"
                        else (0, 180, 255),
                    )

                if visible_prediction is not None and state == "testing":
                    put_text(
                        canvas,
                        f"Prediccion: X={visible_prediction[0]:.3f} "
                        f"Y={visible_prediction[1]:.3f}",
                        (25, screen_height - 55),
                        0.52,
                        (0, 255, 80),
                    )

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if (state == "intro" and key == 32
                        or state == "results" and key in (ord("r"), ord("R"))):
                    evaluation_log = EvaluationLog(model_hash, screen_size)
                    results = None
                    state = "testing"
                    target_index = 0
                    target_started_at = time.monotonic()
                    smoothed_prediction = None
                    last_prediction_time = None

    finally:
        try:
            if evaluation_log is not None:
                evaluation_log.finish(completed=False)
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
