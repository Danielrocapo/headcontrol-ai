"""Mide el efecto de cambios leves de postura sobre un objetivo fijo."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import random
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gaze_feedback_experiment import (
    ERROR_FIELDS, POSE_FIELDS, PREDICTION_FIELDS, prepare_screen, quit_pressed, write_table,
)
from gaze_live_test import (
    BASE_DIR, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, EXPECTED_FEATURES,
    FACE_TASK_PATH, MODEL_PATH, SETTLE_SECONDS, SMOOTHING_ALPHA, TARGET_SECONDS, TARGETS,
    draw_target, extract_model_features, get_screen_size, prediction_errors, put_centered_text,
)


WINDOW = "HeadControl AI - experimento de postura"
POINT_IDS = (2, 4, 5, 6, 7, 8, 9)
POSTURES = {
    "neutral": ("Neutral", "Cabeza en posicion natural."),
    "left": ("Izquierda", "Gira ligeramente la cabeza hacia tu izquierda."),
    "right": ("Derecha", "Gira ligeramente la cabeza hacia tu derecha."),
    "up": ("Arriba", "Levanta ligeramente la barbilla."),
    "down": ("Abajo", "Baja ligeramente la barbilla."),
}
MODES = ("raw", "smoothed")
ERROR_AXES = ("total", "x", "y")
TRIAL_FIELDS = ("trial_id", "repetition", "point_id", "posture")
FRAME_FIELDS = (
    *TRIAL_FIELDS, "timestamp_ms", "timestamp_utc", "elapsed_target_s", "phase",
    "target_x", "target_y", *PREDICTION_FIELDS, *ERROR_FIELDS, *EXPECTED_FEATURES,
    *POSE_FIELDS, "detection_status", "valid_detection", "pose_status",
    "camera_width", "camera_height", "screen_width", "screen_height",
)
ABS_ERRORS = tuple(
    f"{mode}_abs_error_{axis}_px" for mode in MODES for axis in ERROR_AXES
)
TRIAL_METRICS = (
    *(f"{field}_mean" for field in ABS_ERRORS),
    *(f"{field}_median" for field in ABS_ERRORS + ERROR_FIELDS + tuple(EXPECTED_FEATURES) + POSE_FIELDS),
)
DELTA_FIELDS = (
    *(f"{mode}_shift_{axis}_px" for mode in MODES for axis in ("x", "y", "total")),
    *(f"delta_{field}" for field in ABS_ERRORS + tuple(EXPECTED_FEATURES) + POSE_FIELDS),
)


def make_plan(repetitions, seed):
    points = list(POINT_IDS)
    random.Random(seed).shuffle(points)
    postures = list(POSTURES)
    plan = []
    for repetition in range(1, repetitions + 1):
        reverse = repetition % 2 == 0
        for point in reversed(points) if reverse else points:
            offset = (points.index(point) + (repetition - 1) // 2) % len(postures)
            order = postures[offset:] + postures[:offset]
            for posture in reversed(order) if reverse else order:
                plan.append(dict(trial_id=len(plan) + 1, repetition=repetition,
                                 point_id=point, posture=posture))
    return plan


def pose_angles(result):
    blank = (None, None, None)
    if not result.face_landmarks:
        return blank, "no_face"
    if not result.facial_transformation_matrixes:
        return blank, "matrix_missing"
    matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-5) or matrix[2, 3] >= 0):
        return blank, "invalid_matrix"

    # Quita la escala uniforme. Cambia de Y arriba/Z hacia la camara a Y abajo/Z al fondo.
    u, scale, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if scale[-1] <= 1e-6 or scale[0] / scale[-1] > 1.05 or np.linalg.det(rotation) <= 0:
        return blank, "invalid_rotation"
    axes = np.diag([1.0, -1.0, -1.0])
    pitch, yaw, roll = cv2.RQDecomp3x3(axes @ rotation @ axes)[0]
    if not np.isfinite((yaw, pitch, roll)).all():
        return blank, "invalid_rotation"
    return (yaw, pitch, roll), "ok_approximate"


def show(canvas, delay=1):
    cv2.imshow(WINDOW, canvas)
    return cv2.waitKey(delay) & 0xFF


def wait_to_start(canvas):
    while True:
        key = show(canvas, 30)
        if key == 32:
            return True
        if quit_pressed(key) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False


def capture_trials(camera, detector, model, screen, plan, output, trials):
    width, height = screen
    if not wait_to_start(prepare_screen(screen, [
        "EXPERIMENTO DE POSTURA",
        "Mira siempre el punto amarillo mientras cambias la postura.",
        "Manten la distancia, la camara y la iluminacion.",
        "Mueve la cabeza poco, sin acercarte ni inclinarla hacia un hombro.",
        "ESPACIO = comenzar   Q / ESC = guardar y salir",
    ])):
        return False

    last_timestamp = 0
    for trial in plan:
        target = TARGETS[trial["point_id"] - 1]
        background = np.zeros((height, width, 3), dtype=np.uint8)
        draw_target(background, target, width, height)
        cue = background.copy()
        title, instruction = POSTURES[trial["posture"]]
        scale = min(width / 1280, height / 720)
        put_centered_text(cue, f"P{trial['point_id']}  |  {title}  |  "
                          f"Prueba {trial['trial_id']}/{len(plan)}",
                          (width // 2, int(height * 0.028)), int(width * 0.9), 0.62 * scale)
        put_centered_text(cue, f"{instruction}  ESPACIO cuando estes listo.",
                          (width // 2, int(height * 0.975)), int(width * 0.94), 0.65 * scale)
        if not wait_to_start(cue) or quit_pressed(show(background)):
            return False

        started = time.monotonic()
        smoothed = None
        samples = []
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
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_timestamp
            )
            features = extract_model_features(result.face_landmarks[0]) if result.face_landmarks else None
            raw = None
            if features is not None:
                prediction = model.predict(features.reshape(1, -1))[0]
                if np.isfinite(prediction).all():
                    raw = prediction
                    clipped = np.clip(raw, 0.0, 1.0)
                    smoothed = clipped if smoothed is None else (
                        SMOOTHING_ALPHA * clipped + (1.0 - SMOOTHING_ALPHA) * smoothed
                    )
            valid = raw is not None
            smooth_sample = smoothed if valid else None
            if valid:
                status = "valid"
            elif features is not None:
                status = "invalid_prediction"
            else:
                status = "invalid_eyes" if result.face_landmarks else "no_face"

            angles, pose_status = pose_angles(result)
            phase = "settling" if elapsed < SETTLE_SECONDS else "measurement"
            row = {
                **trial, **dict(zip(FRAME_FIELDS[len(TRIAL_FIELDS):], (
                    last_timestamp, captured_utc, elapsed, phase, *target,
                    *(raw if valid else (None, None)), *(smooth_sample if valid else (None, None)),
                    *prediction_errors(raw, target, screen), *prediction_errors(smooth_sample, target, screen),
                    *(features if features is not None else (None,) * len(EXPECTED_FEATURES)),
                    *angles, status, int(valid), pose_status, frame_width, frame_height, width, height,
                ), strict=True)),
            }
            output.writerow(row)
            if phase == "measurement":
                samples.append(row)
            if quit_pressed(show(background)) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
        trials.append(summarize_trial(trial, samples))
    return True


def summarize_trial(trial, rows):
    valid = [row for row in rows if row["valid_detection"]]
    pose = [row for row in rows if row["pose_status"] == "ok_approximate"]
    eyes = [row for row in rows if row[EXPECTED_FEATURES[0]] is not None]
    joint = [row for row in valid if row["pose_status"] == "ok_approximate"]
    summary = dict(trial, frames=len(rows), valid_detections=len(valid),
                   eye_frames=len(eyes), pose_frames=len(pose), joint_frames=len(joint))
    for group, fields in ((valid, ERROR_FIELDS), (eyes, EXPECTED_FEATURES), (pose, POSE_FIELDS)):
        if group:
            for field in fields:
                summary[f"{field}_median"] = float(np.median([row[field] for row in group]))
    if valid:
        for field in ERROR_FIELDS:
            errors = np.abs([row[field] for row in valid])
            absolute = field.replace("_error_", "_abs_error_")
            summary[f"{absolute}_mean"] = float(errors.mean())
            summary[f"{absolute}_median"] = float(np.median(errors))
    if joint:
        for field in POSE_FIELDS + ERROR_FIELDS:
            summary[f"joint_{field}_median"] = float(np.median([row[field] for row in joint]))
    return summary


def paired_trials(trials):
    neutral = {(row["repetition"], row["point_id"]): row for row in trials
               if row["posture"] == "neutral"}
    pairs = []
    for row in trials:
        baseline = neutral.get((row["repetition"], row["point_id"]))
        if row["posture"] == "neutral" or baseline is None:
            continue
        pair = {field: row[field] for field in TRIAL_FIELDS}
        for field in ABS_ERRORS + tuple(EXPECTED_FEATURES) + POSE_FIELDS:
            key = f"{field}_median"
            if key in row and key in baseline:
                pair[f"delta_{field}"] = row[key] - baseline[key]
        for prefix in ("", "joint_"):
            for mode in MODES:
                for axis in ("x", "y"):
                    key = f"{prefix}{mode}_error_{axis}_px_median"
                    if key in row and key in baseline:
                        pair[f"{prefix}{mode}_shift_{axis}_px"] = row[key] - baseline[key]
        for field in POSE_FIELDS:
            key = f"joint_{field}_median"
            if key in row and key in baseline:
                pair[f"joint_delta_{field}"] = row[key] - baseline[key]
        for mode in MODES:
            x, y = (pair.get(f"{mode}_shift_{axis}_px") for axis in ("x", "y"))
            if x is not None and y is not None:
                pair[f"{mode}_shift_total_px"] = float(np.hypot(x, y))
        pairs.append(pair)
    return pairs


def correlations(pairs):
    for angle in POSE_FIELDS:
        for mode in MODES:
            for axis in ("x", "y"):
                angle_key = f"joint_delta_{angle}"
                shift_key = f"joint_{mode}_shift_{axis}_px"
                samples = [(row[angle_key], row[shift_key]) for row in pairs
                           if angle_key in row and shift_key in row]
                value = None
                if len(samples) >= 3 and np.all(np.ptp(samples, axis=0) > 1e-9):
                    value = float(np.corrcoef(samples, rowvar=False)[0, 1])
                yield dict(angle=angle, prediction=mode, axis=axis, pairs=len(samples), pearson_r=value)


def aggregate_trials(trials):
    groups = defaultdict(list)
    for row in trials:
        groups[(row["posture"], row["point_id"])].append(row)
        groups[(row["posture"], "all")].append(row)
    for posture in POSTURES:
        for point in ("all", *POINT_IDS):
            rows = groups[(posture, point)]
            result = dict(posture=posture, point_id=point, trials=len(rows))
            for key in ("frames", "valid_detections", "eye_frames", "pose_frames", "joint_frames"):
                result[key] = sum(row[key] for row in rows)
            for field in TRIAL_METRICS:
                values = [row[field] for row in rows if field in row]
                if values:
                    result[field] = float(np.mean(values) if field.endswith("_mean") else np.median(values))
            yield result


def plot_comparison(folder, summary, pairs):
    global_rows = [row for row in summary if row["point_id"] == "all"]
    labels = [label for label, _ in POSTURES.values()]
    positions = np.arange(len(labels))
    figure = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, mode, label in zip(figure.subplots(1, 2, sharey=True), MODES, ("Directa", "Suavizada")):
        for offset, axis, color in ((-0.18, "x", "#1864ab"), (0.18, "y", "#b44918")):
            values = [row.get(f"{mode}_abs_error_{axis}_px_median", np.nan) for row in global_rows]
            ax.bar(positions + offset, values, 0.33, label=axis.upper(), color=color)
        ax.set(title=label, xticks=positions, xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    figure.suptitle("Error absoluto mediano por postura y eje")
    figure.supylabel("Píxeles")
    figure.savefig(folder / "errores_por_postura.png", dpi=160)

    figure = Figure(figsize=(11, 6), layout="constrained")
    FigureCanvasAgg(figure)
    labels = labels[1:]
    maps = []
    for mode in MODES:
        grid = np.full((len(POINT_IDS), len(labels)), np.nan)
        for i, point in enumerate(POINT_IDS):
            for j, posture in enumerate(list(POSTURES)[1:]):
                key = f"{mode}_shift_total_px"
                values = [row[key] for row in pairs
                          if row["point_id"] == point and row["posture"] == posture and key in row]
                if values:
                    grid[i, j] = np.median(values)
        maps.append(grid)
    maximum = max((np.max(grid[np.isfinite(grid)]) for grid in maps if np.isfinite(grid).any()), default=1)
    axes = figure.subplots(1, 2, sharey=True)
    for ax, grid, label in zip(axes, maps, ("Directa", "Suavizada")):
        mesh = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1, maximum), aspect="auto")
        ax.set(title=label, xticks=range(len(labels)), xticklabels=labels,
               yticks=range(len(POINT_IDS)), yticklabels=[f"P{i}" for i in POINT_IDS])
        for i, j in np.ndindex(grid.shape):
            value = grid[i, j]
            color = "white" if value > maximum * 0.65 else "#222222"
            ax.text(j, i, f"{value:.0f}" if np.isfinite(value) else "—",
                    ha="center", va="center", color=color)
    figure.colorbar(mesh, ax=axes, label="Píxeles")
    figure.suptitle("Desplazamiento mediano respecto a neutral en el mismo punto")
    figure.savefig(folder / "sensibilidad_por_punto.png", dpi=160)


def write_report(folder, trials, plan, model_hash, screen, seed, complete):
    counts = ("frames", "valid_detections", "eye_frames", "pose_frames", "joint_frames")
    joint_fields = tuple(f"joint_{field}_median" for field in POSE_FIELDS + ERROR_FIELDS)
    joint_deltas = (*(f"joint_delta_{field}" for field in POSE_FIELDS),
                    *(f"joint_{mode}_shift_{axis}_px" for mode in MODES for axis in ("x", "y")))
    pairs = paired_trials(trials)
    summary = list(aggregate_trials(trials))
    relations = list(correlations(pairs))
    write_table(folder / "plan.csv", TRIAL_FIELDS, plan)
    write_table(folder / "fijaciones.csv", (*TRIAL_FIELDS, *counts, *TRIAL_METRICS, *joint_fields), trials)
    write_table(folder / "resumen.csv", ("posture", "point_id", "trials", *counts, *TRIAL_METRICS), summary)
    write_table(folder / "comparacion_neutral.csv", (*TRIAL_FIELDS, *DELTA_FIELDS, *joint_deltas), pairs)
    write_table(folder / "correlaciones.csv", ("angle", "prediction", "axis", "pairs", "pearson_r"), relations)
    lines = [
        f"Estado: {'completado' if complete else 'interrumpido'}",
        f"Modelo: {MODEL_PATH}", f"SHA-256: {model_hash}", f"Monitor: {screen[0]} x {screen[1]}",
        f"Orden reproducible: semilla {seed}, {len(trials)}/{len(plan)} fijaciones completadas.",
        "Posturas indicadas por el protocolo. Los angulos medidos sirven para verificar la diferencia real.",
        "Angulos aproximados desde la matriz de MediaPipe en imagen espejada.",
        "Yaw+ hacia la izquierda, pitch+ hacia abajo y roll+ horario en la imagen.",
        f"Metricas sobre detecciones nuevas tras {SETTLE_SECONDS:.1f} s de fijacion.",
        "Mismo peso por fijacion. El resumen promedia las medias y toma la mediana de las medianas.",
        "X+ = derecha e Y+ = abajo. Los errores absolutos separan magnitud y direccion.",
        "Comparacion neutral: cambios dentro del mismo punto y repeticion.",
        "Correlaciones descriptivas de cambios pareados respecto a neutral, sobre fotogramas "
        "con prediccion y pose validas. Los pares de una sesion no son observaciones independientes.",
        "",
        "Posturas | Directa total/X/Y media-mediana (px) | Suavizada total/X/Y media-mediana (px) "
        "| Yaw/pitch/roll medianos | poses validas",
    ]
    for row in summary:
        if row["point_id"] != "all":
            continue
        errors = []
        for mode in MODES:
            stats = []
            for axis in ERROR_AXES:
                prefix = f"{mode}_abs_error_{axis}_px"
                mean, median = row.get(f"{prefix}_mean"), row.get(f"{prefix}_median")
                stats.append("--" if mean is None else f"{mean:.1f}/{median:.1f}")
            errors.append("  ".join(stats))
        angles = " / ".join(f"{row[key + '_median']:+.1f}" if key + "_median" in row else "--"
                            for key in POSE_FIELDS)
        lines.append(f"{POSTURES[row['posture']][0]} | {' | '.join(errors)} | "
                     f"{angles} | {row['pose_frames']}/{row['frames']}")

    for mode, name in zip(MODES, ("directa", "suavizada")):
        lines.extend(["", f"Prediccion {name}: sensibilidad por punto frente a neutral (mediana en px)"])
        ranking = []
        for point in POINT_IDS:
            values = [row[f"{mode}_shift_total_px"] for row in pairs
                      if row["point_id"] == point and f"{mode}_shift_total_px" in row]
            if values:
                ranking.append((np.median(values), point, len(values)))
        lines.extend(f"P{point}: {value:.1f} px ({count} pares)" for value, point, count in sorted(ranking, reverse=True))
        lines.extend(["", "Mediana del desplazamiento absoluto por eje respecto a neutral (px)"])
        for posture in list(POSTURES)[1:]:
            shifts = []
            for axis in ("x", "y"):
                key = f"{mode}_shift_{axis}_px"
                values = [abs(row[key]) for row in pairs if row["posture"] == posture and key in row]
                shifts.append(f"{np.median(values):.1f}" if values else "--")
            lines.append(f"{POSTURES[posture][0]} | X {shifts[0]} | Y {shifts[1]}")

    lines.extend(["", "Cambios medianos de las entradas al comparar con neutral",
                  " | ".join(("Postura", *EXPECTED_FEATURES))])
    for posture in list(POSTURES)[1:]:
        changes = []
        for field in EXPECTED_FEATURES:
            key = f"delta_{field}"
            values = [row[key] for row in pairs if row["posture"] == posture and key in row]
            changes.append(f"{np.median(values):+.4f}" if values else "--")
        lines.append(" | ".join((POSTURES[posture][0], *changes)))

    lines.extend(["", "Correlaciones de cambios de pose con desplazamiento por eje"])
    for row in relations:
        value = f"{row['pearson_r']:+.3f}" if row["pearson_r"] is not None else "--"
        lines.append(f"{row['angle']} | {row['prediction']} {row['axis'].upper()} | "
                     f"r {value} | {row['pairs']} pares")
    (folder / "informe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_comparison(folder, summary, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=2, help="Minimo 2 (por defecto: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla del orden inicial de puntos.")
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions debe ser al menos 2.")
    try:
        model_bytes = MODEL_PATH.read_bytes()
        bundle = joblib.load(BytesIO(model_bytes))
        model = bundle["model"]
        if (model is None or bundle.get("feature_names") != EXPECTED_FEATURES
                or bundle.get("target_names") != ["target_x", "target_y"]):
            raise ValueError("Las entradas y salidas no coinciden con la prueba en vivo.")
        model_hash = sha256(model_bytes).hexdigest()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"No se pudo abrir el modelo: {error}\n")
    if not FACE_TASK_PATH.is_file():
        parser.exit(1, f"No se encuentra el detector facial: {FACE_TASK_PATH}\n")

    screen = get_screen_size()
    plan = make_plan(args.repetitions, args.seed)
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
            num_faces=1, output_face_blendshapes=False, output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        folder = BASE_DIR / f"gaze_postura_{datetime.now():%Y%m%d_%H%M%S_%f}"
        folder.mkdir()
        trials = []
        complete = False
        try:
            with (folder / "fotogramas.csv").open("x", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, FRAME_FIELDS)
                writer.writeheader()
                with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                    complete = capture_trials(camera, detector, model, screen, plan, writer, trials)
        finally:
            write_report(folder, trials, plan, model_hash, screen, args.seed, complete)
            print(f"Resultados: {folder}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
