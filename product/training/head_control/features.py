"""Contrato de cámara y características de HeadControlNet v1."""

from datetime import datetime, timezone
from hashlib import sha256
from inspect import getsourcefile
from pathlib import Path
import sys
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path[:0] = [str(EXPERIMENTS / "phase4_2"), str(EXPERIMENTS / "phase4")]
from validate_head_pose import open_camera
from capture_phase4_validation import (
    CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, FACE_TASK_HASH, FACE_TASK_PATH, analysis_rgb, pose_angles,
)

VERSION = "headcontrolnet-face-v1"
RAW = ("yaw_deg", "pitch_deg", "roll_deg", "nose_x", "nose_y", "chin_x", "chin_y")
FEATURES = ("yaw_rel_deg", "pitch_rel_deg", "roll_rel_deg",
            "nose_x_rel", "nose_y_rel", "chin_x_rel", "chin_y_rel")
MIN_EYE_SPAN = 25


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def relative(values, baseline):
    delta = np.asarray(values) - baseline
    delta[:3] = (delta[:3] + 180) % 360 - 180
    return dict(zip(FEATURES, map(float, delta)))


def geometry(landmarks):
    xy = np.asarray([(landmarks[i].x * CAMERA_WIDTH, landmarks[i].y * CAMERA_HEIGHT)
                     for i in (33, 263, 1, 152)], dtype=float)
    if not np.isfinite(xy).all():
        return None, None
    eyes = xy[:2] if xy[0, 0] <= xy[1, 0] else xy[1::-1]
    axis = eyes[1] - eyes[0]
    span = float(np.linalg.norm(axis))
    if span < MIN_EYE_SPAN or axis[0] <= 0:
        return None, span
    horizontal = axis / span
    vertical = np.array((-horizontal[1], horizontal[0]))
    offsets = (xy[2:] - eyes.mean(axis=0)) / span
    return (offsets @ np.stack((horizontal, vertical), axis=1)).ravel(), span


class CameraFeatures:
    def __init__(self, stack):
        if sha256(FACE_TASK_PATH.read_bytes()).hexdigest() != FACE_TASK_HASH:
            raise RuntimeError("SHA-256 inesperado del Face Landmarker.")
        self.camera = open_camera(stack)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=.5, min_face_presence_confidence=.5,
            min_tracking_confidence=.5, output_facial_transformation_matrixes=True,
        )
        self.detector = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(options))
        self.previous = None
        self.timestamp = -1
        scripts = {Path(getsourcefile(fn)).resolve() for fn in (open_camera, pose_angles, analysis_rgb, geometry)}
        self.metadata = dict(
            contract=VERSION, features=list(FEATURES), raw_fields=list(RAW), face_task_sha256=FACE_TASK_HASH,
            camera_index=CAMERA_INDEX, camera_backend=self.camera.getBackendName(),
            camera_width=CAMERA_WIDTH, camera_height=CAMERA_HEIGHT, mirrored=True,
            colors="BGR -> RGB", mode="VIDEO", num_faces=1,
            thresholds=dict(face_detection=.5, face_presence=.5, tracking=.5),
            eye_landmarks=[33, 263], nose_landmark=1, chin_landmark=152, min_eye_span_px=MIN_EYE_SPAN,
            opencv=cv2.__version__, mediapipe=mp.__version__, numpy=np.__version__,
            source_sha256={p.relative_to(EXPERIMENTS.parents[1]).as_posix(): sha256(p.read_bytes()).hexdigest()
                           for p in sorted(scripts)},
        )

    def read(self):
        ok, frame = self.camera.read()
        now = monotonic()
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        row = dict(timestamp_utc=utc(), timestamp_ms=self.timestamp, valid_detection=0, face_detected=0)
        if not ok or frame is None:
            return now, dict(row, status="camera_read_failed"), None, ()
        height, width = frame.shape[:2]
        if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            raise RuntimeError(f"Resolución efectiva {width} × {height}. Se requiere 1280 × 720.")
        row.update(camera_width=width, camera_height=height)
        rgb = analysis_rgb(frame)
        if self.previous is not None and np.array_equal(frame, self.previous):
            return now, dict(row, status="duplicate_frame"), rgb, ()
        self.previous = frame.copy()
        result = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                self.timestamp)
        landmarks = result.face_landmarks[0] if result.face_landmarks else ()
        angles, status = pose_angles(result)
        row.update(face_detected=int(bool(landmarks)), pose_status=status)
        if status != "ok_approximate":
            return now, dict(row, status=status), rgb, landmarks
        if len(landmarks) <= 263:
            return now, dict(row, status="missing_landmarks"), rgb, landmarks
        offsets, span = geometry(landmarks)
        row["interocular_px"] = span
        if offsets is None:
            return now, dict(row, status="invalid_eye_geometry"), rgb, landmarks
        row.update(zip(RAW, map(float, (*angles, *offsets))))
        return now, dict(row, status="valid", valid_detection=1), rgb, landmarks
