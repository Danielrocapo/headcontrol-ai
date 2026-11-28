"""Audita capturas oculares y reconstruye las fijaciones sin mezclar sesiones."""

import csv
from collections import Counter
from hashlib import sha256
from io import StringIO
import json
from math import isfinite
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EYES = ("eye_a_x", "eye_a_y", "eye_a_open", "eye_b_x", "eye_b_y", "eye_b_open")
POSE = ("head_yaw_deg", "head_pitch_deg", "head_roll_deg")
TARGETS = ("target_x", "target_y")
FILES = {
    "historical_837": "gaze_dataset_20261123_142451.csv",
    "historical_1651": "gaze_dataset_20261123_150016.csv",
    "head_pose": "gaze_postura_20261123_190502_834178/fotogramas.csv",
}


def nonfinite(row, key):
    try:
        return not isfinite(float(row[key]))
    except (TypeError, ValueError):
        return True


def inspect_dataset(name):
    path = ROOT / "prueba_ojos" / FILES[name]
    contents = path.read_bytes()
    reader = csv.DictReader(StringIO(contents.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    is_pose = name == "head_pose"
    split_key = "repetition" if is_pose else "round"
    held = "2" if is_pose else "3"
    features = EYES + POSE if is_pose else EYES
    grouping = ("trial_id",) if is_pose else ("round", "point_id")
    required = set((*grouping, split_key, "point_id", "timestamp_ms", *features, *TARGETS))
    if is_pose:
        required.update(("phase", "valid_detection", "pose_status", "posture", "camera_width",
                         "camera_height", "screen_width", "screen_height"))
    if not rows or not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Faltan filas o columnas requeridas en {path}.")

    groups = {}
    eligible = []
    for line, row in enumerate(rows, 2):
        group = ":".join(row[key] for key in grouping)
        point = int(row["point_id"])
        target = tuple(float(row[key]) for key in TARGETS)
        grid = (0.08, 0.50, 0.92)
        if not 1 <= point <= 9 or target != (grid[(point - 1) % 3], grid[(point - 1) // 3]):
            raise ValueError(f"Objetivo incompatible con P{point} en la línea {line} de {path}.")
        metadata = (point, target, row[split_key], row.get("posture", "unspecified"))
        if group in groups and groups[group] != metadata:
            raise ValueError(f"La fijación {group} mezcla etiquetas o particiones en {path}.")
        groups[group] = metadata
        if is_pose and (row["phase"], row["valid_detection"], row["pose_status"]) != (
            "measurement", "1", "ok_approximate",
        ):
            continue
        if any(nonfinite(row, key) for key in features):
            raise ValueError(f"Una fila elegible contiene entradas inválidas en {path}:{line}.")
        camera = (row.get("camera_width"), row.get("camera_height"))
        if is_pose and camera != ("1280", "720"):
            raise ValueError(f"La captura de pose no tiene geometría 1280 × 720 en {path}:{line}.")
        screen = (int(row["screen_width"]), int(row["screen_height"])) if is_pose else (1920, 1080)
        if min(screen) <= 0:
            raise ValueError(f"Resolución inválida en {path}:{line}.")
        eligible.append(dict(row, fixation=group, source_line=line, split="reserved" if row[split_key] == held
                             else "development", screen_width=screen[0], screen_height=screen[1]))
    timestamps = [row["timestamp_ms"] for row in eligible]
    if any(not row[key] for row in eligible for key in grouping) or len(timestamps) != len(set(timestamps)):
        raise ValueError(f"Faltan identificadores de fijación o hay timestamps repetidos en {path}.")

    available = tuple(key for key in reader.fieldnames if key.startswith(("eye_", "head_")))
    breakdown = {}
    for split in ("development", "reserved"):
        part = [row for row in eligible if row["split"] == split]
        breakdown[split] = dict(frames=len(part), fixations=len({row["fixation"] for row in part}))
    report = dict(
        path=path.relative_to(ROOT).as_posix(), sha256=sha256(contents).hexdigest(), frames=len(rows),
        fixations=len(groups), capture_files=1, independent_sessions_confirmed=False,
        grouping=grouping, partition_key=split_key, reserved_value=held, splits=breakdown,
        known_camera_sizes=sorted({f"{r['camera_width']}x{r['camera_height']}" for r in rows}) if is_pose else [],
        pixel_reference="screen_width/height por fila" if is_pose else "1920x1080 del protocolo histórico",
        features=available, all_columns=reader.fieldnames,
        feature_missing={key: sum(not row[key] for row in rows) for key in available},
        feature_invalid={key: sum(nonfinite(row, key) for row in rows) for key in available},
        full_duplicates=len(rows) - len({tuple(row.values()) for row in rows}),
        repeated_timestamps=len(rows) - len({row["timestamp_ms"] for row in rows}),
        repeated_feature_targets=len(rows) - len({tuple(row[key] for key in (*features, *TARGETS)) for row in rows}),
        frames_by_point=dict(Counter(int(row["point_id"]) for row in rows)),
        eligible_by_point=dict(Counter(int(row["point_id"]) for row in eligible)),
        fixations_by_point=dict(Counter(values[0] for values in groups.values())),
        eligible_by_posture=dict(Counter(row["posture"] for row in eligible)) if is_pose else {},
        profile=("Ojos del extractor en vivo y grados de la matriz MediaPipe espejada. "
                 "Geometría 16:9 registrada. Verificar paridad multiplataforma antes de producción."
                 if is_pose else "Mismas fórmulas oculares. Resolución efectiva sin registrar. "
                 "Los proxies 2D de pose y el roll en radianes son incompatibles con features-v2."),
        use=("experimental_only" if name != "historical_837" else "audit_only"),
    )
    return eligible, report


if __name__ == "__main__":
    print(json.dumps({name: inspect_dataset(name)[1] for name in FILES}, indent=2, ensure_ascii=False))
