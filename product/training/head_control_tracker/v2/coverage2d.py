"""Interpolación de cobertura en un abanico triangular, en píxeles."""

from itertools import combinations
from pathlib import Path

import numpy as np

from calibrate_user import atomic_json, read_json
from calibration_math import score
from features9 import VERSION

PROFILES = Path(__file__).resolve().parent / "coverage2d_profiles"
SCHEMA = "tracker-coverage2d-v1"
MAX_GAIN = 2.5
MIN_SEPARATION_PX = 8.
VALIDATION_TOLERANCE_PX = .5
ANCHORS = {
    "neutral": (.5, .5),
    "superior izquierda": (0., 0.), "arriba": (.5, 0.), "superior derecha": (1., 0.),
    "derecha": (1., .5), "inferior derecha": (1., 1.), "abajo": (.5, 1.),
    "inferior izquierda": (0., 1.), "izquierda": (0., .5),
}
HOLDOUT = {"interior superior izquierdo": (.25, .25), "interior superior derecho": (.75, .25),
           "interior inferior derecho": (.75, .75), "interior inferior izquierdo": (.25, .75)}


class CoverageMap:
    def __init__(self, originals, size):
        self.scale = np.asarray(size, dtype=float) - 1
        if self.scale.shape != (2,) or not np.all(self.scale > 0):
            raise ValueError("La pantalla debe tener al menos dos píxeles por eje.")
        raw = np.array([originals[k] for k in ANCHORS], dtype=float) * self.scale
        if raw.shape != (9, 2) or not np.isfinite(raw).all():
            raise ValueError("Se necesitan nueve anclas RAW finitas.")
        targets = np.rint(np.array(list(ANCHORS.values())) * self.scale)
        self.center, self.target_center = raw[0], targets[0]
        outer, mapped = raw[1:] - self.center, targets[1:] - self.target_center
        source = np.stack((outer, np.roll(outer, -1, axis=0)), axis=-1)
        destination = np.stack((mapped, np.roll(mapped, -1, axis=0)), axis=-1)
        determinants = np.linalg.det(source)
        dots = np.einsum("ij,ij->i", source[:, :, 0], source[:, :, 1])
        if (np.any(determinants <= 0) or
                not np.isclose(np.arctan2(determinants, dots).sum(), 2*np.pi, atol=1e-8, rtol=0)):
            raise ValueError("Las anclas RAW no delimitan regiones en orden alrededor del neutral.")
        self.inverse = np.linalg.inv(source)
        self.destination = destination
        self.inner_gain = float(np.linalg.svd(destination @ self.inverse, compute_uv=False)[:, 0].max())
        # Fuera del contorno, u+v se normaliza a 1. La ganancia máxima allí
        # se acota por la longitud del lado de salida por la mayor norma de fila.
        sides = np.linalg.norm(destination[:, :, 1]-destination[:, :, 0], axis=1)
        self.outer_gain = float(np.max(sides * np.linalg.norm(self.inverse, axis=2).max(axis=1)))
        if max(self.inner_gain, self.outer_gain) > MAX_GAIN:
            raise ValueError(f"La malla amplificaría el ruido hasta {max(self.inner_gain,self.outer_gain):.2f}× "
                             f"(límite {MAX_GAIN:g}× en píxeles).")

    def apply(self, xy):
        delta = np.asarray(xy) * self.scale - self.center
        weights = self.inverse @ delta
        # El abanico cubre todo el plano. En una frontera ambos sectores dan la misma salida.
        sector = int(np.argmax(weights.min(axis=1)))
        weights = np.maximum(weights[sector], 0.)
        weights /= max(1., weights.sum())
        point = self.target_center + self.destination[sector] @ weights
        return np.clip(point / self.scale, 0., 1.)


def fit(samples, size):
    scale = np.array(size) - 1
    clouds = {key: np.asarray(samples[key]["samples"], dtype=float) * scale for key in ANCHORS}
    if any(a.ndim != 2 or a.shape[1] != 2 or len(a) < 12 or not np.isfinite(a).all() for a in clouds.values()):
        raise ValueError("Cada ancla necesita al menos 12 predicciones válidas.")
    medians = {key: np.median(a, axis=0) for key, a in clouds.items()}
    for a, b in combinations(ANCHORS, 2):
        delta = medians[b] - medians[a]
        distance = np.linalg.norm(delta)
        if (distance < MIN_SEPARATION_PX or np.percentile(clouds[a] @ (delta / distance), 90) >=
                np.percentile(clouds[b] @ (delta / distance), 10)):
            raise ValueError(f"RAW no distingue bien «{a}» de «{b}» ({distance:.1f} px entre medianas "
                             "o intervalos P10–P90 solapados).")
    originals = {key: (value / scale).tolist() for key, value in medians.items()}
    return CoverageMap(originals, size), originals


def validate(mapping, checks):
    targets = [p["target"] for p in checks]
    before = [p["original"] for p in checks]
    after = [np.median([mapping.apply(x) for x in p["samples"]], axis=0).tolist() for p in checks]
    raw = score(before, targets, mapping.scale)
    corrected = score(after, targets, mapping.scale)
    accepted = all(corrected[k] <= raw[k] + VALIDATION_TOLERANCE_PX for k in ("mean_px", "p90_px"))
    return dict(accepted=accepted, raw=raw, corrected=corrected, corrected_points=after)


def publish(folder):
    atomic_json(PROFILES / "active.json", {"profile": folder.name})


def load_active(identity, size, camera):
    selection = PROFILES / "active.json"
    if not selection.is_file():
        return None
    run = read_json(selection)["profile"]
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Selección de cobertura 2D inválida.")
    profile = read_json(PROFILES / run / "profile.json")
    if (profile["schema"] != SCHEMA or profile["feature_contract"] != VERSION
            or tuple(profile["screen_size"]) != tuple(size) or profile["validation"]["accepted"] is not True):
        raise ValueError("Perfil 2D no aprobado o incompatible con el monitor.")
    if any(profile["model"][key] != identity[key] for key in ("weights_sha256", "onnx_sha256", "normalizer_sha256")):
        raise ValueError("El perfil de cobertura 2D pertenece a otro modelo.")
    keys = ("source_sha256", "face_task_sha256", "camera_index", "camera_width", "camera_height", "mirrored")
    if any(profile["camera"][key] != camera[key] for key in keys):
        raise ValueError("El perfil de cobertura 2D pertenece a otro extractor o cámara.")
    return CoverageMap(profile["originals"], size)
