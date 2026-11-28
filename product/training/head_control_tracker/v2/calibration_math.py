"""Corrección afín de salida y evaluación por fijación."""

import numpy as np

GRID = [(x, y) for y in (.08, .50, .92) for x in (.08, .50, .92)]
HELD_OUT = [(x, y) for y in (.29, .71) for x in (.29, .71)] + [
    (.50, .29), (.50, .71), (.29, .50), (.71, .50)]


def affine(predictions, matrix):
    values = np.asarray(predictions, dtype=float)
    return values @ matrix[:2] + matrix[2]


def fit(points):
    inputs = np.array([p["original"] for p in points])
    labels = np.array([p["manual"] if p["manual"] is not None else p["target"] for p in points])
    design = np.column_stack((inputs, np.ones(len(inputs))))
    if not np.isfinite(design).all() or not np.isfinite(labels).all():
        raise ValueError("Las coordenadas de ajuste no son finitas.")
    matrix, _, rank, singular = np.linalg.lstsq(design, labels, rcond=None)
    if rank != 3:
        raise ValueError("Las predicciones de los puntos de ajuste no abarcan dos dimensiones.")
    return matrix, float(singular[0] / singular[-1])


def score(predicted, targets, scale):
    predicted, targets = np.asarray(predicted), np.asarray(targets)
    error = (predicted - targets) * scale
    length = np.linalg.norm(error, axis=1)
    return dict(mean_px=float(length.mean()), median_px=float(np.median(length)),
                p90_px=float(np.percentile(length, 90)), mae_x_px=float(np.abs(error[:, 0]).mean()),
                mae_y_px=float(np.abs(error[:, 1]).mean()),
                outside=int(np.any((predicted < 0) | (predicted > 1), axis=1).sum()), points=len(targets))


def compare(points, matrix, scale):
    before = np.array([p["original"] for p in points])
    expected = np.array([p["target"] for p in points])
    after = np.array([np.median(affine(p["samples"], matrix), axis=0) for p in points])
    raw, corrected = score(before, expected, scale), score(after, expected, scale)
    return raw, corrected, corrected["mean_px"] < raw["mean_px"]


class Smoother:
    def __init__(self, seconds):
        self.seconds = seconds
        self.reset()

    def reset(self):
        self.value = self.time = None

    def step(self, target, now):
        if self.value is None or not self.seconds:
            self.value = np.array(target, dtype=float, copy=True)
        else:
            gain = -np.expm1(-max(now - self.time, 0.) / self.seconds)
            self.value += gain * (target - self.value)
        self.time = now
        return self.value
