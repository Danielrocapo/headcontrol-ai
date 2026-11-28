"""Control relativo por velocidad a partir de las coordenadas RAW de Tracker D."""

from dataclasses import dataclass
from math import expm1, sqrt

import numpy as np


DEFAULT_DEADZONE = .025
FULL_DEFLECTION = .25
MAX_INTERVAL_S = .05
MIN_NEUTRAL_SAMPLES = 12


def stable_neutral(samples, deadzone=DEFAULT_DEADZONE):
    points = np.asarray(samples, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < MIN_NEUTRAL_SAMPLES
            or not np.isfinite(points).all()):
        raise ValueError("Se necesitan al menos 12 predicciones RAW válidas.")
    center = np.median(points, axis=0)
    if np.any(np.percentile(np.abs(points - center), 90, axis=0) > deadzone):
        raise ValueError("El P90 de la desviación RAW supera la zona muerta configurada.")
    return center


@dataclass
class RelativeStep:
    output: np.ndarray
    deviation: np.ndarray
    velocity: np.ndarray
    clipped: np.ndarray
    state: str


class RelativeCursor:
    def __init__(self, tau, deadzone, speed, max_step):
        self.tau = tau
        self.deadzone = deadzone
        self.speed = speed
        # Reserva el margen de la cuantización y del resto subpíxel.
        self.max_dt = min(MAX_INTERVAL_S, (max_step - sqrt(2)) / speed)
        self.reset()

    def reset(self):
        self.last_stamp = None
        self.ready = False
        self.velocity = np.zeros(2)
        self.remainder = np.zeros(2)

    def step(self, raw, neutral, current, stamp, scale):
        if neutral is None or not np.isfinite(raw).all() or not np.isfinite(neutral).all():
            self.reset()
            return None
        if np.any((current < 0) | (current > scale)):
            self.reset()
            return None
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return None
        dt = 0. if self.last_stamp is None else min(stamp - self.last_stamp, self.max_dt)
        self.last_stamp = stamp
        deviation = np.asarray(raw) - neutral
        still = np.abs(deviation) <= self.deadzone
        if still.all():
            self.velocity.fill(0.)
            self.remainder.fill(0.)
            self.ready = True
            return RelativeStep(current, deviation, self.velocity.copy(), np.zeros(2, bool), "neutral")
        if not self.ready:
            return RelativeStep(current, deviation, self.velocity.copy(), np.zeros(2, bool), "esperando neutral")
        command = np.clip((np.abs(deviation) - self.deadzone) / (FULL_DEFLECTION - self.deadzone), 0., 1.)
        command *= np.sign(deviation)
        command *= self.speed / max(1., np.linalg.norm(command))
        alpha = -expm1(-dt / self.tau) if self.tau else 1.
        self.velocity += alpha * (command - self.velocity)
        self.velocity[still] = 0.
        self.remainder[still] = 0.
        proposed = current + self.velocity * dt + self.remainder
        bounded = np.clip(proposed, 0, scale)
        output = np.rint(bounded).astype(int)
        self.remainder = bounded - output
        return RelativeStep(output, deviation, self.velocity.copy(), proposed != bounded, "movimiento")
