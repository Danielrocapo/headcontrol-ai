"""Captura ciega por sesión o prueba técnica de cámara de 10 segundos."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import random
import re
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from phase4_models import EYES, HERE, MODELS, POSE, ROOT, prediction_engine

sys.path.insert(0, str(ROOT / "prueba_ojos"))
from gaze_head_pose_experiment import POSTURES, pose_angles
from gaze_live_test import (
    CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, EXPECTED_FEATURES, FACE_TASK_PATH, SETTLE_SECONDS,
    TARGET_SECONDS, TARGETS, draw_target, extract_model_features, get_screen_size, put_centered_text,
)


WINDOW = "HeadControl AI - validacion independiente"
POINTS = {**{f"P{i}": xy for i, xy in enumerate(TARGETS, 1)},
          "I1": (0.29, 0.29), "I2": (0.71, 0.29), "I3": (0.29, 0.71), "I4": (0.71, 0.71)}
PREDICTION_FIELDS = tuple(f"{model}_{axis}" for model in MODELS for axis in ("x", "y", "sha256"))
FIELDS = (
    "session_id", "trial_id", "target_id", "target_x", "target_y", "posture", "timestamp_ms",
    "timestamp_utc", "elapsed_s", "phase", "status", "pose_status", *EYES, *POSE, *PREDICTION_FIELDS,
    "camera_width", "camera_height", "screen_width", "screen_height",
)
FACE_TASK_HASH = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def session_plan(slot):
    rng = random.Random(42004)
    points, postures = list(POINTS), list(POSTURES)
    rng.shuffle(points)
    rng.shuffle(postures)
    plan = [(point, posture) for i, point in enumerate(points)
            for j, posture in enumerate(postures) if (i + j) % 3 + 1 == slot]
    rng.shuffle(plan)
    return plan


def analysis_rgb(frame):
    height, width = frame.shape[:2]
    k = min(width // 16, height // 9)
    if not k:
        raise ValueError("Resolución de cámara insuficiente.")
    w, h = 16 * k, 9 * k
    x, y = (width - w) // 2, (height - h) // 2
    mirrored = cv2.flip(frame, 1)
    cropped = mirrored[y:y + h, x:x + w]
    if (w, h) != (CAMERA_WIDTH, CAMERA_HEIGHT):
        cropped = cv2.resize(cropped, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def key(canvas, delay=1):
    cv2.imshow(WINDOW, canvas)
    pressed = cv2.waitKey(delay) & 0xFF
    return 27 if pressed in (ord("q"), ord("Q")) else pressed


def ready(canvas):
    while True:
        pressed = key(canvas, 30)
        if pressed in (27, 32):
            return pressed == 32


def capture(camera, detector, predict, models, screen, plan, session_id, writer, metadata, smoke):
    w, h = screen
    last_ts, previous = 0, None
    statuses = Counter()
    metadata["status_counts"] = statuses
    for index, (point, posture) in enumerate(plan, 1):
        target = POINTS[point]
        if not smoke:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            draw_target(canvas, target, w, h)
            cue = canvas.copy()
            name, instruction = POSTURES[posture]
            scale = min(w / 1280, h / 720)
            put_centered_text(cue, f"{index}/{len(plan)}  |  {name}  |  {point}",
                              (w // 2, int(h * 0.025)), int(w * 0.9), 0.66 * scale)
            put_centered_text(cue, f"{instruction}  ESPACIO = listo  Q / ESC = salir",
                              (w // 2, int(h * 0.972)), int(w * 0.94), 0.64 * scale)
            if not ready(cue) or key(canvas) == 27:
                return
        started = time.monotonic()
        duration = 10.0 if smoke else TARGET_SECONDS
        while time.monotonic() - started < duration:
            ok, frame = camera.read()
            now = time.monotonic()
            if not ok:
                statuses["camera_read_failed"] += 1
                return
            if now - started >= duration:
                break
            camera_h, camera_w = frame.shape[:2]
            size = [camera_w, camera_h]
            if metadata["camera_size"] is None:
                metadata["camera_size"] = size
            if size != metadata["camera_size"]:
                raise ValueError(f"La resolución efectiva cambió: {metadata['camera_size']} → {size}.")
            last_ts = max(last_ts + 1, int(now * 1000))
            row = dict(session_id=session_id, trial_id=index, target_id=point, target_x=target[0],
                       target_y=target[1], posture=posture, timestamp_ms=last_ts, elapsed_s=now - started,
                       timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       phase="smoke" if smoke else "settling" if now - started < SETTLE_SECONDS else "measurement",
                       camera_width=camera_w, camera_height=camera_h, screen_width=w, screen_height=h)
            for name in MODELS:
                row[f"{name}_sha256"] = models[name]["sha256"]
            if previous is not None and np.array_equal(previous, frame):
                row["status"] = "duplicate_frame"
            else:
                rgb = analysis_rgb(frame)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_ts)
                angles, pose_status = pose_angles(result)
                row["pose_status"] = pose_status
                eyes = extract_model_features(result.face_landmarks[0]) if result.face_landmarks else None
                if eyes is None:
                    row["status"] = "invalid_eyes" if result.face_landmarks else "no_face"
                elif pose_status != "ok_approximate":
                    row["status"] = "invalid_pose"
                else:
                    values = predict(np.r_[eyes, angles])
                    row["status"] = "valid" if all(values[name] is not None for name in MODELS) else "prediction_discarded"
                    row.update(zip(EYES + POSE, np.r_[eyes, angles], strict=True))
                    for name in MODELS:
                        if values[name] is not None:
                            row[f"{name}_x"], row[f"{name}_y"] = values[name]
                            metadata["finite_predictions"][name] += 1
            previous = frame.copy()
            statuses[row["status"]] += 1
            writer.writerow(row)
            if not smoke and key(canvas) == 27:
                return
        metadata["completed_trials"].append(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--session", type=int, choices=(1, 2, 3))
    parser.add_argument("--series", default="validacion_01")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.series):
        parser.error("La serie solo admite letras, números, guión y guión bajo.")
    if not args.smoke and args.series == "smoke":
        parser.error("El nombre smoke está reservado para la prueba técnica.")
    if tuple(EXPECTED_FEATURES) != EYES or sha256(FACE_TASK_PATH.read_bytes()).hexdigest() != FACE_TASK_HASH:
        parser.error("El detector o el extractor ocular no coincide con la referencia.")
    screen = get_screen_size()
    plan = [("P5", "neutral")] if args.smoke else session_plan(args.session)
    series = HERE / "captures" / ("smoke" if args.smoke else args.series)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    session_id = f"smoke_{stamp}" if args.smoke else f"session_{args.session}_{stamp}"
    previous = [json.loads(p.read_text(encoding="utf-8")) for p in series.glob("session_*/session.json")]
    complete = [p for p in previous if p["complete"]]
    if not args.smoke and any(p["slot"] == args.session for p in complete):
        parser.error("Ese número de sesión ya está completo en la serie. Usa una serie nueva para repetir.")
    if any(p["screen_size"] != list(screen) for p in complete):
        parser.error("La resolución del monitor difiere de la serie existente.")
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not camera.isOpened():
            raise RuntimeError("No se pudo abrir la cámara.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_facial_transformation_matrixes=True,
        )
        with prediction_engine() as (models, predict):
            if any(p["models"] != models for p in complete):
                raise ValueError("Los modelos difieren de los utilizados en la serie.")
            if not args.smoke:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            folder = series / session_id
            folder.mkdir(parents=True)
            metadata = dict(session_id=session_id, slot=args.session, complete=False, smoke=args.smoke,
                            screen_size=list(screen), camera_size=complete[0]["camera_size"] if complete else None,
                            face_task_sha256=FACE_TASK_HASH, models=models, plan=plan,
                            completed_trials=[], finite_predictions={name: 0 for name in MODELS})
            try:
                with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector, (
                    folder / "fotogramas.csv"
                ).open("x", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, FIELDS)
                    writer.writeheader()
                    capture(camera, detector, predict, models, screen, plan, session_id, writer, metadata, args.smoke)
            finally:
                metadata["complete"] = len(metadata["completed_trials"]) == len(plan)
                (folder / "session.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
                print(f"Captura: {folder}\nEstados: {metadata.get('status_counts', {})}\n"
                      f"Predicciones finitas: {metadata['finite_predictions']}", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
