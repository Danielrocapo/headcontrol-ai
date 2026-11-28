"""Valida el protocolo de postura sin ejecutar modelos de mirada."""

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from report import CONDITIONS, NEAR_LIMIT_DEG, TRAINING_CAMERA, coverage, make_report, training_pitch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase4"))
from capture_phase4_validation import (
    CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, FACE_TASK_HASH, FACE_TASK_PATH, analysis_rgb, pose_angles,
)
from phase4_models import verified

WINDOW = "HeadControl AI - Fase 4.2"
FIELDS = (
    "trial_id", "repetition", "condition", "timestamp_utc", "timestamp_ms", "elapsed_s",
    "face_detected", "valid_detection", "status", "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "relative_pitch_deg", "pitch_coverage", "camera_width", "camera_height",
)
ORDERS = (
    ("neutral", "up_light", "up_moderate"),
    ("up_light", "up_moderate", "neutral"),
    ("up_moderate", "neutral", "up_light"),
    ("neutral", "up_moderate", "up_light"),
    ("up_light", "neutral", "up_moderate"),
)


def open_camera(stack):
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)
    stack.callback(camera.release)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la camara de fase 4.")
    for setting, value in ((cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH),
                           (cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT),
                           (cv2.CAP_PROP_BUFFERSIZE, 1)):
        camera.set(setting, value)
    return camera


class PoseCapture:
    def __init__(self, detector, training, preview):
        self.detector = detector
        self.training = training
        self.preview = preview
        self.previous = None
        self.timestamp = 0
        self.baseline = None

    def sample(self, camera):
        ok, frame = camera.read()
        now = time.monotonic()
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        row = dict(timestamp_ms=self.timestamp, timestamp_utc=datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"), valid_detection=0)
        if not ok:
            return now, dict(row, status="camera_read_failed"), None, ()
        row["camera_height"], row["camera_width"] = frame.shape[:2]
        if self.previous is not None and np.array_equal(self.previous, frame):
            return now, dict(row, status="duplicate_frame"), analysis_rgb(frame) if self.preview else None, ()
        self.previous = frame.copy()
        rgb = analysis_rgb(frame)
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                self.timestamp)
        landmarks = result.face_landmarks[0] if result.face_landmarks else ()
        angles, status = pose_angles(result)
        row.update(status=status, face_detected=int(bool(landmarks)))
        if status == "ok_approximate":
            yaw, pitch, roll = angles
            row.update(valid_detection=1, head_yaw_deg=yaw, head_pitch_deg=pitch, head_roll_deg=roll,
                       pitch_coverage=coverage(pitch, self.training))
            if self.baseline is not None:
                row["relative_pitch_deg"] = pitch - self.baseline
        return now, row, rgb, landmarks

    def display(self, row, rgb, landmarks, intro, instruction, band=None):
        if not self.preview:
            return -1
        canvas = np.full((690, 960, 3), (24, 24, 24), np.uint8)
        if rgb is not None:
            preview = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (960, 540))
            if landmarks:
                for idx in (1, 33, 133, 152, 263, 362):
                    point = landmarks[idx]
                    cv2.circle(preview, (round(point.x * 960), round(point.y * 540)), 4, (0, 220, 240), -1)
            canvas[:540] = preview
        good = (100, 238, 165)
        warning = (100, 215, 255)
        white = (235, 235, 235)
        baseline = f"{self.baseline:+.1f}" if self.baseline is not None else "--"
        angles = "Pitch --  |  Yaw --  |  Roll --  |  Delta neutral --"
        status = f"Sin pose valida ({row['status']})"
        color = warning
        if row["valid_detection"]:
            relative = row.get("relative_pitch_deg")
            delta = f"{relative:+.1f}" if relative is not None else "--"
            angles = (f"Pitch {row['head_pitch_deg']:+.1f}  |  Yaw {row['head_yaw_deg']:+.1f}"
                      f"  |  Roll {row['head_roll_deg']:+.1f}  |  Delta neutral {delta} grados")
            color = good if band is None or (relative is not None and band[0] <= relative <= band[1]) else warning
            labels = {"inside": "dentro", "near_limit": "dentro, cerca del limite", "outside": "fuera"}
            status = f"Rostro y pose validos  |  Pitch {labels[row['pitch_coverage']]} del rango entrenado"
        elif row.get("face_detected"):
            status = f"Rostro detectado, pose invalida ({row['status']})"
        lo, mid, hi = (self.training[k] for k in ("min", "median", "max"))
        lines = (
            (intro, white), (angles, color), (status, color),
            (f"Entrenamiento {lo:+.1f} a {hi:+.1f} grados (mediana {mid:+.1f})  |  Neutral ref. {baseline}", white),
            (instruction, white),
        )
        for y, (label, shade) in enumerate(lines):
            cv2.putText(canvas, label, (18, 563 + 27 * y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, shade, 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        pressed = cv2.waitKey(1) & 0xFF
        if pressed in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return 27
        return pressed

    def record(self, camera, writer, trial, duration, rest=True):
        key = -1
        condition = CONDITIONS.get(trial["condition"])
        band = condition[1] if condition else None
        if trial["condition"] == "baseline":
            title = "Referencia neutral   |   Cabeza comoda y mirada al centro"
        elif condition:
            title = (f"Serie {trial['repetition']}/5   |   {condition[0]}   |   "
                     f"Delta orientativo {band[0]:+g} a {band[1]:+g} grados")
        else:
            title = "Prueba tecnica de camara"
        while rest and key != 32:
            _, row, rgb, landmarks = self.sample(camera)
            if row["status"] == "camera_read_failed":
                raise RuntimeError("Se perdio la lectura de la camara antes de grabar.")
            key = self.display(row, rgb, landmarks, title, "Descansa y ajusta la pose. ESPACIO inicia. Q/ESC sale.", band)
            if key == 27:
                return False, []
        started = time.monotonic()
        pitches = []
        while True:
            now, row, rgb, landmarks = self.sample(camera)
            elapsed = now - started
            if elapsed >= duration:
                return True, pitches
            writer.writerow(dict(row, **trial, elapsed_s=round(elapsed, 6)))
            if row["status"] == "camera_read_failed":
                raise RuntimeError("Se perdio la lectura de la camara durante la grabacion.")
            if row["valid_detection"]:
                pitches.append(row["head_pitch_deg"])
            if self.display(row, rgb, landmarks, title, f"Grabando   {duration - elapsed:.1f} s restantes. Q/ESC sale.",
                            band) == 27:
                return False, pitches


def capture(folder, training, smoke, preview):
    meta = dict(mode="smoke" if smoke else "protocol", training_pitch=training, baseline_pitch_deg=None,
                completed_trials=[], complete=False, face_task_sha256=FACE_TASK_HASH,
                coverage_margin_deg=NEAR_LIMIT_DEG, training_camera=list(TRAINING_CAMERA),
                requested_camera=[CAMERA_WIDTH, CAMERA_HEIGHT], seconds_per_trial=5,
                baseline_seconds=0 if smoke else 3)
    plan = [dict(trial_id=1, repetition=1, condition="smoke")] if smoke else [
        dict(trial_id=(number - 1) * 3 + index, repetition=number, condition=condition)
        for number, order in enumerate(ORDERS, 1) for index, condition in enumerate(order, 1)
    ]
    meta["plan"] = plan
    folder.mkdir(parents=True)
    try:
        verified(FACE_TASK_PATH, FACE_TASK_HASH)
        with ExitStack() as stack:
            camera = open_camera(stack)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, output_facial_transformation_matrixes=True,
                num_faces=1, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(options))
            if preview:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, 960, 690)
                stack.callback(cv2.destroyAllWindows)
            capture = PoseCapture(detector, training, preview=preview)
            file = stack.enter_context((folder / "fotogramas.csv").open("x", encoding="utf-8", newline=""))
            writer = csv.DictWriter(file, FIELDS)
            writer.writeheader()
            if not smoke:
                baseline = dict(trial_id=0, repetition=0, condition="baseline")
                done, pitches = capture.record(camera, writer, baseline, meta["baseline_seconds"])
                if not done:
                    return
                if len(pitches) < 10:
                    raise RuntimeError("Referencia neutral sin 10 detecciones validas. Reinicia con el rostro visible.")
                capture.baseline = meta["baseline_pitch_deg"] = float(np.median(pitches))
            for trial in plan:
                done, _ = capture.record(camera, writer, trial, meta["seconds_per_trial"], rest=preview)
                file.flush()
                if not done:
                    return
                meta["completed_trials"].append(trial["trial_id"])
            meta["complete"] = True
    finally:
        with (folder / "session.json").open("x", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, ensure_ascii=False, allow_nan=False)
        if (folder / "fotogramas.csv").is_file():
            make_report(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Graba cinco segundos para probar la camara.")
    mode.add_argument("--analyze", type=Path, metavar="CARPETA", help="Reanaliza una captura de fase 4.2.")
    parser.add_argument("--preview", action="store_true", help="En --smoke, muestra la camara y espera ESPACIO.")
    args = parser.parse_args()
    if args.preview and not args.smoke:
        parser.error("--preview solo se usa con --smoke. El protocolo completo ya muestra la camara.")
    if args.analyze:
        make_report(args.analyze)
        return
    training, _ = training_pitch()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = HERE / "captures" / f"{'smoke' if args.smoke else 'pose'}_{stamp}"
    print(f"Salida de fase 4.2: {folder}", flush=True)
    capture(folder, training, args.smoke, preview=not args.smoke or args.preview)


if __name__ == "__main__":
    main()
