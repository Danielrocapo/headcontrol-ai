"""Perfil de alcance independiente de la MLP y de la calibración afín."""

from pathlib import Path

import numpy as np

from calibrate_user import atomic_json, read_json
from features9 import VERSION

PROFILES = Path(__file__).resolve().parent / "reach_profiles"
MAX_GAIN = 2.
POINTS = (
    ("neutral", (.5, .5)), ("izquierda", (0., .5)), ("derecha", (1., .5)),
    ("arriba", (.5, 0.)), ("abajo", (.5, 1.)),
)


class ReachMap:
    def __init__(self, knots, *, source=None, neutral_reference=None):
        self.knots = np.asarray(knots, dtype=float)
        if self.knots.shape != (2, 3) or not np.isfinite(self.knots).all():
            raise ValueError("Se necesitan tres anclas finitas para cada eje.")
        spans = np.diff(self.knots, axis=1)
        for axis, direction, span in zip(("X", "X", "Y", "Y"),
                                         ("izquierda", "derecha", "arriba", "abajo"), spans.flat):
            if span <= 0:
                raise ValueError(f"El ancla {direction} no está en el lado esperado del neutral ({axis}).")
            if span < .5 / MAX_GAIN:
                raise ValueError(f"El ancla {direction} exigiría ampliar {axis} {(.5/span):.2f}×. "
                                 f"El máximo es {MAX_GAIN:g}× por tramo.")
        self.gains = .5 / spans
        self.source = source
        self.neutral_reference = neutral_reference

    def apply(self, xy):
        return np.array([np.interp(value, knots, (0., .5, 1.)) for value, knots in zip(xy, self.knots)])

    def reached_sides(self, xy):
        x, y = xy
        return (x <= self.knots[0, 0], x >= self.knots[0, 2],
                y <= self.knots[1, 0], y >= self.knots[1, 2])

    @classmethod
    def from_points(cls, originals):
        return cls([[originals["izquierda"][0], originals["neutral"][0], originals["derecha"][0]],
                    [originals["arriba"][1], originals["neutral"][1], originals["abajo"][1]]])


def set_active(profile):
    PROFILES.mkdir(exist_ok=True)
    if profile is not None:
        relative = profile.resolve().relative_to(PROFILES.resolve()).as_posix()
    else:
        relative = None
    atomic_json(PROFILES / "active.json", {"profile": relative})


def active_reach(identity, size, camera):
    selection = PROFILES / "active.json"
    if not selection.is_file():
        return None
    name = read_json(selection)["profile"]
    if name is None:
        return None
    path = (PROFILES / name).resolve()
    if not path.is_relative_to(PROFILES.resolve()):
        raise ValueError("La ruta del perfil queda fuera de los perfiles de alcance.")
    return load_reach(path, identity, size, camera)


def load_reach(path, identity, size, camera):
    profile = read_json(path)
    if (profile["schema"] != "tracker-reach-v1" or tuple(profile["screen_size"]) != tuple(size)
            or profile["feature_contract"] != VERSION):
        raise ValueError("Perfil de alcance incompatible con este monitor o contrato.")
    for key in ("weights_sha256", "onnx_sha256", "normalizer_sha256"):
        if profile["model"][key] != identity[key]:
            raise ValueError("El perfil de alcance pertenece a otro modelo.")
    for key in ("source_sha256", "face_task_sha256", "camera_index", "camera_width", "camera_height", "mirrored"):
        if profile["camera"][key] != camera[key]:
            raise ValueError("El perfil de alcance pertenece a otro extractor o cámara.")
    return ReachMap(profile["knots"], source=path, neutral_reference=profile.get("neutral_reference"))


def compare_neutral(saved, current):
    angles = (np.array(current["raw"][:3]) - saved["raw"][:3] + 180) % 360 - 180
    center = np.array(current["center_px"]) - saved["center_px"]
    ratio = current["interocular_px"] / saved["interocular_px"]
    return angles, center, (ratio - 1) * 100
