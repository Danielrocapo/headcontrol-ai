"""Control temporal independiente del reconocimiento de clics."""

from dataclasses import dataclass
from math import isfinite


def blink_limits(reading, calibration):
    source, _ = reading
    if calibration is not None and calibration.source == source:
        # La protección empieza antes del umbral que confirma un guiño.
        onset = tuple(op + .35 * (cl - op) for op, cl in zip(calibration.opened, calibration.closed))
        return onset, calibration.opened
    return ((.45, .45), (.30, .30)) if source == "blendshapes" else ((.85, .85), (.82, .82))


@dataclass(frozen=True)
class BlinkStatus:
    holding: bool = False
    resumed: bool = False
    state: str = "Ojos abiertos"


class BlinkGuard:
    def __init__(self, recovery=.18):
        self.recovery = recovery
        self.reset()

    def reset(self):
        self.status = BlinkStatus()
        self.last = self.source = self.open_since = None

    def update(self, stamp, reading, calibration):
        if stamp is None or reading is None or not all(isfinite(v) for v in reading[1]):
            self.open_since = self.last = None
            self.status = BlinkStatus(self.status.holding, state="Sin lectura ocular")
            return self.status
        if self.last == stamp:
            return self.status
        if self.last is None or stamp - self.last > .12 or self.source != reading[0]:
            self.open_since = None
        self.last, self.source = stamp, reading[0]
        onset, reopened = blink_limits(reading, calibration)
        closed = tuple(score >= limit for score, limit in zip(reading[1], onset))
        if any(closed):
            self.open_since = None
            side = "bilateral" if all(closed) else "izquierdo" if closed[0] else "derecho"
            self.status = BlinkStatus(True, state=f"Cierre {side}")
        elif self.status.holding:
            if any(score > limit for score, limit in zip(reading[1], reopened)):
                self.open_since = None
                self.status = BlinkStatus(True, state="Esperando reapertura")
            else:
                if self.open_since is None:
                    self.open_since = stamp
                elapsed = stamp - self.open_since
                if elapsed < self.recovery:
                    self.status = BlinkStatus(True, state="Estabilizando reapertura")
                else:
                    self.status = BlinkStatus(resumed=True, state="Movimiento recuperado")
        else:
            self.status = BlinkStatus()
        return self.status
