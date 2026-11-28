"""Contrato facial del Tracker de nueve entradas."""

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "head_control"))
from features import CameraFeatures, FEATURES as FEATURES7, RAW, MIN_EYE_SPAN, relative

VERSION = "headcontrolnet-tracker-face-v2"
FEATURES = (*FEATURES7, "face_center_dx", "face_center_dy")
CENTER = ("face_center_x_px", "face_center_y_px")
ANGULAR_SPREAD_MAX = 5.0
CENTER_SPREAD_MAX = .06
SPAN_SPREAD_MAX = .15


def face_center(landmarks, width, height):
    a, b = landmarks[33], landmarks[263]
    return (a.x + b.x) * width / 2, (a.y + b.y) * height / 2


class Camera9(CameraFeatures):
    def __init__(self, stack):
        super().__init__(stack)
        self.metadata.update(contract=VERSION, features=list(FEATURES),
                             raw_fields=[*RAW, *CENTER, "interocular_px"],
                             axes="mirrored camera, X right, Y down", reference_scale="neutral_interocular_median_px")
        self.metadata["source_sha256"][
            "product/training/head_control_tracker/v2/features9.py"] = sha256(Path(__file__).read_bytes()).hexdigest()

    def read(self):
        now, row, rgb, landmarks = super().read()
        if row["valid_detection"]:
            center = face_center(landmarks, row["camera_width"], row["camera_height"])
            if not np.isfinite(center).all():
                row.update(valid_detection=0, status="invalid_face_center")
            else:
                row.update(zip(CENTER, center))
        return now, row, rgb, landmarks


def neutral_reference(rows):
    valid = [row for row in rows if row["valid_detection"]]
    if len(valid) < 12:
        raise ValueError("Se necesitan 12 detecciones válidas para la referencia neutral.")
    data = np.array([[r[k] for k in (*RAW, *CENTER, "interocular_px")] for r in valid], dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Referencia neutral con valores no finitos.")
    med = np.median(data, axis=0)
    span = float(med[-1])
    if span < MIN_EYE_SPAN:
        raise ValueError("Distancia interocular neutral insuficiente.")
    spread = np.ptp(np.percentile(data, (10, 90), axis=0), axis=0)
    if (np.max(spread[:3]) > ANGULAR_SPREAD_MAX or
            np.max(spread[7:9]) / span > CENTER_SPREAD_MAX or spread[-1] / span > SPAN_SPREAD_MAX):
        raise ValueError("Referencia neutral inestable.")
    return dict(raw=med[:7].tolist(), center_px=med[7:9].tolist(), interocular_px=span,
                count=len(valid), angle_p90_p10_deg=spread[:3].tolist(),
                center_p90_p10_px=spread[7:9].tolist(), span_p90_p10_px=float(spread[-1]))


def vector(row, ref):
    values = relative([row[k] for k in RAW], ref["raw"])
    for key, source, origin in zip(FEATURES[-2:], CENTER, ref["center_px"]):
        values[key] = (row[source] - origin) / ref["interocular_px"]
    return values
