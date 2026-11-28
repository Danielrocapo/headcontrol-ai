"""Compara la mirada con y sin el punto verde, usando el modelo existente."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gaze_live_test import (
    BASE_DIR, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, EXPECTED_FEATURES,
    FACE_TASK_PATH, MODEL_PATH, SETTLE_SECONDS, SMOOTHING_ALPHA, TARGET_SECONDS,
    TARGETS, draw_prediction, draw_target, extract_model_features,
    get_screen_size, prediction_errors, put_centered_text,
)


DATASET_PATH = BASE_DIR / "gaze_dataset_20261123_150016.csv"
WINDOW = "HeadControl AI - experimento de retroalimentacion"
POSE_FIELDS = ("head_yaw_deg", "head_pitch_deg", "head_roll_deg")
OPEN_FIELDS = ("eye_a_open", "eye_b_open")
PREDICTION_FIELDS = (
    "raw_prediction_x", "raw_prediction_y", "smoothed_prediction_x", "smoothed_prediction_y",
)
ERROR_FIELDS = tuple(
    f"{mode}_error_{axis}_px" for mode in ("raw", "smoothed") for axis in ("total", "x", "y")
)
RANGE_FIELDS = tuple(f"{name}_training_range" for name in EXPECTED_FEATURES)
FRAME_FIELDS = (
    "timestamp_ms", "timestamp_utc", "repetition", "block", "condition", "point_id",
    "elapsed_target_s", "phase", "target_x", "target_y", *PREDICTION_FIELDS, *ERROR_FIELDS,
    "valid_detection", "reused_prediction", *EXPECTED_FEATURES, *RANGE_FIELDS,
    *POSE_FIELDS, "pose_status", "pose_rmse_px", "pose_rmse_eye_span",
    "camera_width", "camera_height", "screen_width", "screen_height",
)

# Vertices del rostro canonico de MediaPipe, en el orden de los indices.
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
POSE_INDICES = (1, 10, 33, 61, 152, 168, 263, 291)
POSE_POINTS = np.array([
    (0, -1.126865, 7.475604), (0, 8.261778, 4.481535),
    (-4.445859, 2.663991, 3.173422), (-2.456206, -4.342621, 4.283884),
    (0, -9.403378, 4.264492), (0, 3.271027, 5.236015),
    (4.445859, 2.663991, 3.173422), (2.456206, -4.342621, 4.283884),
], dtype=np.float64) * (1, -1, -1)
MIN_EYE_SPAN_PX = 24
MAX_POSE_RMSE_FRACTION = 0.08


def estimate_pose(landmarks, width, height):
    blank = (None, None, None)
    if landmarks is None:
        return blank, "no_face", None, None
    if len(landmarks) <= max(POSE_INDICES):
        return blank, "missing_landmarks", None, None

    pixels = np.array([(landmarks[i].x, landmarks[i].y) for i in POSE_INDICES]) * (width, height)
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels >= (width, height))):
        return blank, "landmarks_outside_frame", None, None
    eye_span = np.linalg.norm(pixels[2] - pixels[6])
    if eye_span < MIN_EYE_SPAN_PX:
        return blank, "face_too_small", None, None

    # Intrinsecos aproximados. La comparacion es relativa, en la imagen espejada.
    matrix = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                      dtype=np.float64)
    try:
        ok, rotation, translation = cv2.solvePnP(POSE_POINTS, pixels, matrix, None,
                                                flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            return blank, "pnp_failed", None, None
        orientation, _ = cv2.Rodrigues(rotation)
        depths = (POSE_POINTS @ orientation.T)[:, 2] + translation[2, 0]
        if np.any(depths <= 0) or orientation[2, 2] <= 0:
            return blank, "implausible_pose", None, None
        projected, _ = cv2.projectPoints(POSE_POINTS, rotation, translation, matrix, None)
        rmse = np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - pixels) ** 2, axis=1)))
        fraction = rmse / eye_span
        if not np.isfinite(fraction):
            return blank, "pnp_failed", None, None
        if fraction > MAX_POSE_RMSE_FRACTION:
            return blank, "high_reprojection_error", rmse, fraction
        pitch, yaw, roll = cv2.RQDecomp3x3(orientation)[0]
        if not np.isfinite([yaw, pitch, roll]).all():
            return blank, "pnp_failed", rmse, fraction
        return (yaw, pitch, roll), "ok_approximate", rmse, fraction
    except cv2.error:
        return blank, "pnp_failed", None, None


def load_training_ranges(dataset_path, bundle):
    samples = defaultdict(list)
    rounds = set(bundle["train_rounds"])
    with dataset_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        needed = {"round", "point_id", "target_x", "target_y", *EXPECTED_FEATURES}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("El CSV no contiene las columnas del entrenamiento.")
        for row in reader:
            try:
                split = float(row["round"])
                point_id = float(row["point_id"])
                target = [float(row["target_x"]), float(row["target_y"])]
                features = [float(row[name]) for name in EXPECTED_FEATURES]
            except (TypeError, ValueError):
                continue
            if not np.isfinite([split, point_id, *target, *features]).all() or int(split) not in rounds:
                continue
            if point_id not in range(1, len(TARGETS) + 1) or tuple(target) != TARGETS[int(point_id) - 1]:
                raise ValueError(f"Objetivo inconsistente en la linea {reader.line_num} del CSV.")
            samples[int(point_id)].append(features)

    if set(samples) != set(range(1, len(TARGETS) + 1)):
        raise ValueError("El CSV debe tener muestras de entrenamiento para los nueve puntos.")
    count = sum(map(len, samples.values()))
    if count != bundle["metrics"]["train_samples"]:
        raise ValueError(f"El CSV contiene {count} muestras de entrenamiento; el modelo indica otra cantidad.")
    return {
        point_id: (len(values), np.min(values, axis=0), np.median(values, axis=0), np.max(values, axis=0))
        for point_id, values in sorted(samples.items())
    }


def range_status(features, reference):
    if features is None:
        return (None,) * len(EXPECTED_FEATURES)
    _, low, _, high = reference
    return np.where(features < low, "below", np.where(features > high, "above", "within"))


def protocol(repetitions, first_condition):
    pair = (first_condition, "B" if first_condition == "A" else "A")
    for repetition in range(1, repetitions + 1):
        order = pair if repetition % 2 else pair[::-1]
        for index, condition in enumerate(order, 1):
            yield repetition, 2 * (repetition - 1) + index, condition


def write_table(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_screen(screen, lines):
    width, height = screen
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for index, line in enumerate(lines):
        put_centered_text(canvas, line, (width // 2, int(height * (0.25 + index * 0.11))),
                          int(width * 0.9), 0.85 * min(width / 1280, height / 720))
    return canvas


def show(canvas, delay=1):
    cv2.imshow(WINDOW, canvas)
    return cv2.waitKey(delay) & 0xFF


def quit_pressed(key):
    return key in (27, ord("q"), ord("Q"))


def run_experiment(camera, landmarker, model, screen, plan, ranges, writer, fixations):
    last_timestamp = 0
    width, height = screen
    for repetition, block, condition in plan:
        intro = prepare_screen(screen, [
            f"Repeticion {repetition}  |  Bloque {block}/{len(plan)}",
            "Mira siempre el objetivo amarillo. No sigas el punto verde.",
            "Manten la distancia, posicion de la camara e iluminacion.",
            "Descansa entre bloques. ESPACIO = comenzar   Q / ESC = salir",
        ])
        while True:
            key = show(intro, 30)
            if quit_pressed(key):
                return False
            if key == 32:
                break

        smoothed = None
        last_valid_at = None
        for point_id, target in enumerate(TARGETS, 1):
            background = np.zeros((height, width, 3), dtype=np.uint8)
            draw_target(background, target, width, height)
            if quit_pressed(show(background)):
                return False
            started = time.monotonic()
            rows = []
            while True:
                ok, frame = camera.read()
                now = time.monotonic()
                if not ok:
                    print("Se perdio la lectura de la camara. Se guardara un informe parcial.")
                    return False
                elapsed = now - started
                if elapsed >= TARGET_SECONDS:
                    break

                captured_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                last_timestamp = max(last_timestamp + 1, int(now * 1000))
                frame_height, frame_width = frame.shape[:2]
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_timestamp
                )
                landmarks = result.face_landmarks[0] if result.face_landmarks else None
                features = extract_model_features(landmarks) if landmarks is not None else None
                raw = None
                if features is not None:
                    prediction = model.predict(features.reshape(1, -1))[0]
                    if np.isfinite(prediction).all():
                        raw = prediction
                        clipped = np.clip(raw, 0, 1)
                        smoothed = clipped if smoothed is None else (
                            SMOOTHING_ALPHA * clipped + (1 - SMOOTHING_ALPHA) * smoothed
                        )
                        last_valid_at = now
                visible = smoothed if last_valid_at is not None and now - last_valid_at < 0.6 else None
                angles, pose_status, rmse, rmse_fraction = estimate_pose(landmarks, frame_width, frame_height)
                phase = "settling" if elapsed < SETTLE_SECONDS else "measurement"
                record = dict(zip(FRAME_FIELDS, (
                    last_timestamp, captured_utc, repetition, block, condition, point_id,
                    elapsed, phase, *target, *(raw if raw is not None else (None, None)),
                    *(visible if visible is not None else (None, None)),
                    *prediction_errors(raw, target, screen), *prediction_errors(visible, target, screen),
                    int(raw is not None), int(raw is None and visible is not None),
                    *(features if features is not None else (None,) * len(EXPECTED_FEATURES)),
                    *range_status(features, ranges[point_id]), *angles, pose_status, rmse, rmse_fraction,
                    frame_width, frame_height, width, height,
                ), strict=True))
                writer.writerow(record)
                if phase == "measurement":
                    rows.append(record)

                canvas = background.copy()
                if condition == "A":
                    draw_prediction(canvas, visible, width, height)
                if quit_pressed(show(canvas)):
                    return False
            fixations[(repetition, block, condition, point_id)] = rows
    return True


def fixation_summary(rows):
    valid = [row for row in rows if row["valid_detection"]]
    eyes = [row for row in rows if row[EXPECTED_FEATURES[0]] is not None]
    poses = [row for row in rows if row["pose_status"] == "ok_approximate"]
    summary = {"frames": len(rows), "valid_detections": len(valid),
               "eye_frames": len(eyes), "pose_frames": len(poses)}
    for sample, fields in ((valid, PREDICTION_FIELDS + ERROR_FIELDS),
                           (eyes, EXPECTED_FEATURES), (poses, POSE_FIELDS)):
        for field in fields:
            if sample:
                values = [row[field] for row in sample]
                summary[f"{field}_median"] = float(np.median(values))
                if field in ERROR_FIELDS:
                    summary[f"{field}_mean"] = float(np.mean(values))
    for field in RANGE_FIELDS:
        if eyes:
            for status in ("below", "above"):
                summary[f"{field}_pct_{status}"] = 100 * sum(row[field] == status for row in eyes) / len(eyes)
    return summary


def summarize_fixations(fixations):
    individual = []
    grouped = defaultdict(list)
    by_repetition = defaultdict(dict)
    for (repetition, block, condition, point_id), rows in fixations.items():
        stats = fixation_summary(rows)
        individual.append(dict(repetition=repetition, block=block, condition=condition,
                               point_id=point_id, **stats))
        grouped[(condition, point_id)].append(stats)
        by_repetition[(repetition, point_id)][condition] = stats

    conditions = []
    count_fields = ("frames", "valid_detections", "eye_frames", "pose_frames")
    for condition in ("A", "B"):
        for point_id in range(1, len(TARGETS) + 1):
            group = grouped[(condition, point_id)]
            aggregate = dict(condition=condition, point_id=point_id, fixations=len(group))
            for field in count_fields:
                aggregate[field] = sum(stats[field] for stats in group)
            for field in sorted({key for stats in group for key in stats} - set(count_fields)):
                values = [stats[field] for stats in group if field in stats]
                aggregate[field] = float(np.median(values))
            conditions.append(aggregate)

    paired = []
    for (repetition, point_id), pair in sorted(by_repetition.items()):
        if set(pair) == {"A", "B"}:
            a, b = pair["A"], pair["B"]
            deltas = {f"{field}_B_minus_A": b[field] - a[field]
                      for field in a.keys() & b.keys() if field not in count_fields}
            paired.append(dict(repetition=repetition, point_id=point_id, **deltas))
    return individual, conditions, paired


def plot_predictions(path, conditions, screen):
    width, height = screen
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    for row, condition in enumerate(("A", "B")):
        for column, mode in enumerate(("raw", "smoothed")):
            ax = axes[row, column]
            name = "Directa" if mode == "raw" else "Suavizada"
            feedback = "con punto verde" if condition == "A" else "sin punto verde"
            ax.set_title(f"{condition} ({feedback}) · {name}")
            for stats in conditions:
                if stats["condition"] != condition:
                    continue
                point_id = stats["point_id"]
                tx, ty = np.array(TARGETS[point_id - 1]) * screen
                ax.scatter(tx, ty, color="#cca000", marker="x")
                ax.annotate(f"P{point_id}", (tx, ty), xytext=(4, 4), textcoords="offset points")
                x = stats.get(f"{mode}_prediction_x_median")
                y = stats.get(f"{mode}_prediction_y_median")
                if x is not None and y is not None:
                    color = "#184a85" if point_id < 7 else "#a63621"
                    ax.plot([tx, x * width], [ty, y * height], color=color, alpha=0.65)
                    ax.scatter(x * width, y * height, color=color, s=21)
            ax.set(xlim=(-0.04 * width, 1.04 * width), ylim=(1.04 * height, -0.04 * height),
                   aspect="equal")
            ax.grid(alpha=0.2)
            ax.label_outer()
    figure.supxlabel("X (px)")
    figure.supylabel("Y (px)")
    figure.suptitle("Objetivo (×) y predicción mediana (●) · P7–P9 en rojo")
    figure.savefig(path, dpi=160)


def write_report(folder, fixations, ranges, plan, screen, model_hash, completed):
    individual, conditions, paired = summarize_fixations(fixations)
    count_fields = ("frames", "valid_detections", "eye_frames", "pose_frames")
    metric_fields = [
        *(f"{field}_median" for field in PREDICTION_FIELDS + ERROR_FIELDS + tuple(EXPECTED_FEATURES) + POSE_FIELDS),
        *(f"{field}_mean" for field in ERROR_FIELDS),
        *(f"{field}_pct_{status}" for field in RANGE_FIELDS for status in ("below", "above")),
    ]
    write_table(folder / "fijaciones.csv", ("repetition", "block", "condition", "point_id",
                                           *count_fields, *metric_fields), individual)
    write_table(folder / "resumen.csv", ("condition", "point_id", "fixations",
                                        *count_fields, *metric_fields), conditions)
    write_table(folder / "comparacion_pareada.csv", (
        "repetition", "point_id", *(f"{field}_B_minus_A" for field in metric_fields),
    ), paired)
    items = (
        dict(point_id=point, feature=name, samples=count, minimum=low[index],
             median=middle[index], maximum=high[index])
        for point, (count, low, middle, high) in ranges.items()
        for index, name in enumerate(EXPECTED_FEATURES)
    )
    write_table(folder / "rangos_entrenamiento.csv",
                ("point_id", "feature", "samples", "minimum", "median", "maximum"), items)
    lines = [
        f"Estado: {'completado' if completed else 'interrumpido'}",
        f"Modelo y SHA-256: {MODEL_PATH} | {model_hash}",
        f"CSV de entrenamiento: {DATASET_PATH}",
        f"Monitor: {screen[0]} x {screen[1]}",
        "Bloques programados: " + " ".join(condition for _, _, condition in plan),
        f"Fijaciones completadas: {len(fixations)}/{len(plan) * len(TARGETS)}",
        f"Protocolo: P1-P9, {TARGET_SECONDS:.1f} s por punto, "
        f"se excluyen los primeros {SETTLE_SECONDS:.1f} s. Alpha del suavizado: {SMOOTHING_ALPHA}.",
        "A: amarillo y verde. B: solo amarillo. X+ derecha; Y+ abajo.",
        "resumen.csv: cada fijacion aporta el mismo peso; se toma la mediana entre fijaciones "
        "de cada estadistico. Los conteos se suman.",
        "comparacion_pareada.csv: diferencia B menos A para el mismo punto y repeticion. "
        "Las celdas quedan vacias si falta una medicion.",
        "Errores: solo detecciones nuevas. Apertura y rangos: fotogramas con entradas oculares. "
        "Postura: fotogramas con ajuste valido, incluso sin deteccion ocular.",
        "Postura aproximada en grados (imagen espejada), rostro canonico MediaPipe, "
        "focal supuesta igual al ancho y distorsion nula. Usar para comparaciones relativas "
        "con la misma camara. Los grados no son comparables con los proxies 2D del CSV original.",
        "Signos de postura en la imagen espejada: yaw+ hacia la izquierda, "
        "pitch+ hacia abajo, roll+ en sentido horario.",
        f"Se omiten angulos sin ajuste valido, con distancia interocular < {MIN_EYE_SPAN_PX} px "
        f"o con RMSE > {MAX_POSE_RMSE_FRACTION:.0%} de esa distancia. Ver pose_status en el CSV.",
        "",
        "P7-P9 | Error total mediano (px), apertura mediana y % bajo el rango del mismo punto",
    ]

    def display(stats, key, decimals=2):
        return f"{stats[key]:.{decimals}f}" if key in stats else "--"

    extra_fields = ("raw_error_total_px_median", "smoothed_error_total_px_median", *(
        f"{field}_median" for field in (*OPEN_FIELDS, *POSE_FIELDS)
    ))
    for stats in conditions:
        if stats["point_id"] < 7:
            continue
        metrics = "  ".join(display(stats, field) for field in extra_fields)
        opening_below = " / ".join(display(stats, f"{field}_training_range_pct_below", 1)
                                   for field in OPEN_FIELDS)
        lines.append(f"P{stats['point_id']} {stats['condition']} | D S abrirA abrirB yaw pitch roll: "
                     f"{metrics} | % bajo rango A/B: {opening_below} | poses: {stats['pose_frames']}")
    (folder / "informe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_predictions(folder / "prediccion_vs_objetivo.png", conditions, screen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=2,
                        help="Numero de pares A/B, al menos 2 (por defecto: 2).")
    parser.add_argument("--first-condition", choices=("A", "B"), default="A")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions debe ser al menos 2.")

    try:
        model_bytes = MODEL_PATH.read_bytes()
        bundle = joblib.load(BytesIO(model_bytes))
        model = bundle.get("model")
        if (model is None or bundle.get("feature_names") != EXPECTED_FEATURES
                or bundle.get("target_names") != ["target_x", "target_y"]):
            raise ValueError("El modelo no tiene las entradas y salidas de la prueba en vivo.")
        model_hash = sha256(model_bytes).hexdigest()
        ranges = load_training_ranges(DATASET_PATH, bundle)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo preparar el experimento: {error}\n")
    if not FACE_TASK_PATH.is_file():
        parser.exit(1, f"No se encuentra el detector facial: {FACE_TASK_PATH}\n")

    screen = get_screen_size()
    plan = list(protocol(args.repetitions, args.first_condition))
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    try:
        if not camera.isOpened():
            parser.exit(1, "No se pudo abrir la camara.\n")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1, output_face_blendshapes=False,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        folder = BASE_DIR / f"gaze_experimento_{datetime.now():%Y%m%d_%H%M%S_%f}"
        folder.mkdir()
        fixations = {}
        completed = False
        try:
            with (folder / "fotogramas.csv").open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, FRAME_FIELDS)
                writer.writeheader()
                with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
                    completed = run_experiment(camera, landmarker, model, screen, plan, ranges, writer, fixations)
        finally:
            write_report(folder, fixations, ranges, plan, screen, model_hash, completed)
            print(f"Resultados: {folder}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
