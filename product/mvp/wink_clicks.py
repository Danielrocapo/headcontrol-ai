"""Lectura ocular y máquina temporal para clics por guiño."""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from blink_guard import blink_limits


BLINK_NAMES = ("eyeBlinkLeft", "eyeBlinkRight")
EYES = ((263, 362, 386, 374, 385, 380), (33, 133, 159, 145, 158, 153))
GESTURES = ("Ambos ojos abiertos", "Guiña el ojo IZQUIERDO", "Cierra ambos ojos (referencia derecha)")
LEFT_GESTURES = (GESTURES[0], GESTURES[1])


def blink_channels(values, mirrored):
    return values[::-1] if mirrored else values


def eye_measurement(categories, landmarks, width, height, mirrored):
    scores = {c.category_name: c.score for c in categories}
    if all(name in scores for name in BLINK_NAMES):
        values = tuple(float(scores[name]) for name in BLINK_NAMES)
        if all(isfinite(value) and 0 <= value <= 1 for value in values):
            return "blendshapes", blink_channels(values, mirrored)
        return None
    if len(landmarks) <= 386:
        return None
    values = []
    for eye in EYES:
        a, b, c, d, e, f = [np.array((landmarks[i].x * width, landmarks[i].y * height)) for i in eye]
        span = np.linalg.norm(a - b)
        if not isfinite(span) or span < 3:
            return None
        values.append(float(1 - (np.linalg.norm(c - d) + np.linalg.norm(e - f)) / (2 * span)))
    if all(isfinite(value) for value in values):
        return "EAR", blink_channels(tuple(values), mirrored)
    return None


@dataclass(frozen=True)
class EyeThresholds:
    source: str
    opened: tuple
    closed: tuple
    open_reference: tuple
    wink_reference: tuple
    right_during_left: float

    def normalize(self, values):
        return tuple((value - low) / (high - low)
                     for value, low, high in zip(values, self.open_reference, self.wink_reference))

    @staticmethod
    def samples(phases, count, source=None):
        if len(phases) != count or any(len(phase) < 12 for phase in phases):
            raise ValueError("Se necesitan al menos 12 lecturas válidas en cada postura ocular.")
        source = phases[0][0][0] if source is None else source
        if any(reading[0] != source for phase in phases for reading in phase):
            raise ValueError("La señal ocular cambió durante la calibración.")
        values = [np.array([reading[1] for reading in phase]) for phase in phases]
        if not all(np.isfinite(phase).all() for phase in values):
            raise ValueError("La calibración contiene señales no finitas.")
        return source, values

    @staticmethod
    def limits(source, low, high, eye):
        gap = high - low
        if gap < (.12 if source == "blendshapes" else .04):
            side = ("izquierdo", "derecho")[eye]
            raise ValueError(f"No se distinguió el ojo {side}. Repite el guiño manteniendo el otro abierto.")
        return float(low + .22 * gap), float(low + .72 * gap)

    @classmethod
    def fit(cls, phases):
        source, values = cls.samples(phases, 3)
        opened, closed, lows, highs = [], [], [], []
        for eye in range(2):
            low = np.percentile(values[0][:, eye], 80)
            high = np.percentile(values[eye+1][:, eye], 20)
            op, cl = cls.limits(source, low, high, eye)
            opened.append(op)
            closed.append(cl)
            lows.append(float(low))
            highs.append(float(high))
        result = cls(source, tuple(opened), tuple(closed), tuple(lows), tuple(highs), opened[1])
        return result.with_right_open_band(values[1])

    def with_right_open_band(self, left_wink):
        other = float(np.percentile(left_wink[:, 1], 90))
        gap = self.closed[1] - self.opened[1]
        cap = self.opened[1] + .30 * gap
        if other > cap:
            raise ValueError("El derecho se cerró durante la calibración izquierda. Repite dejando el derecho abierto.")
        band = min(cap, max(self.opened[1], other + .05 * gap)) if other > self.opened[1] else self.opened[1]
        return replace(self, right_during_left=band)

    def recalibrate_left(self, phases):
        _, (neutral, wink) = self.samples(phases, 2, self.source)
        low, high = float(np.percentile(neutral[:, 0], 80)), float(np.percentile(wink[:, 0], 20))
        op, cl = self.limits(self.source, low, high, 0)
        return replace(self, opened=(op, self.opened[1]), closed=(cl, self.closed[1]),
                       open_reference=(low, self.open_reference[1]),
                       wink_reference=(high, self.wink_reference[1])).with_right_open_band(wink)


@dataclass(frozen=True)
class WinkResult:
    eyes: tuple = ("—", "—")
    holding: bool = False
    confirmed: bool = False
    reopened: bool = False
    bilateral: bool = False
    interrupted: bool = False
    state: str = "Sin lectura"
    notice: str | None = None
    qualified: bool = False
    prior_open_ms: float = 0.


class WinkDetector:
    def __init__(self, duration=.30):
        self.duration = duration
        self.reset()

    def reset(self):
        self.last = self.open_since = self.start = None
        self.held = 0.
        self.prior_open_ms = 0.
        self.closed = [False, False]
        self.seen_onset = [False, False]
        self.primed = False

    def update(self, stamp, reading, thresholds):
        if reading is None or not all(isfinite(v) for v in reading[1]):
            self.reset()
            return WinkResult(interrupted=True, notice="Sin señal ocular válida.")
        if thresholds is None or reading[0] != thresholds.source:
            self.reset()
            return WinkResult(interrupted=True, state="Calibración pendiente" if thresholds is None else "Fuente ocular distinta")
        if self.last is not None and (stamp <= self.last or stamp - self.last > .12):
            self.reset()
            self.last = stamp
            return WinkResult(interrupted=True, notice="Intervalo sin lecturas válidas.")
        self.last = stamp
        values = reading[1]
        opened = [value <= limit for value, limit in zip(values, thresholds.opened)]
        for i, value in enumerate(values):
            if opened[i]:
                self.closed[i] = False
            elif value >= thresholds.closed[i]:
                self.closed[i] = True
        eyes = tuple("cerrado" if closed else "abierto" if op else "transición"
                     for closed, op in zip(self.closed, opened))
        if all(opened):
            confirmed, notice = False, None
            if self.start is not None:
                if self.held < self.duration:
                    notice = f"Izquierdo: duración {self.held*1000:.0f} / {self.duration*1000:.0f} ms."
                elif stamp - self.start > 2.5:
                    notice = "Izquierdo: reapertura fuera de tiempo."
                else:
                    confirmed, notice = True, "Guiño izquierdo confirmado."
            self.start = None
            self.seen_onset = [False, False]
            if self.open_since is None:
                self.open_since = stamp
            self.prior_open_ms = (stamp - self.open_since) * 1000
            self.primed = self.prior_open_ms >= 100.
            return WinkResult(eyes, confirmed=confirmed, reopened=True,
                              state="Listo" if self.primed else "Estabilizando apertura", notice=notice,
                              prior_open_ms=self.prior_open_ms)
        self.open_since = None
        # Un parpadeo se invalida antes de alcanzar el umbral de cierre de un guiño.
        onset, _ = blink_limits(reading, thresholds)
        for i, value in enumerate(values):
            self.seen_onset[i] |= value >= onset[i]
        bilateral = all(self.seen_onset) or all(self.closed)
        if bilateral:
            self.start = None
            self.primed = False
            return WinkResult(eyes, bilateral=True, state="Cierre bilateral", notice="Cierre bilateral: gesto rechazado.")
        left, right = thresholds.normalize(values)
        other_open = not self.closed[1] and values[1] <= thresholds.right_during_left
        dominant = left - right >= (.20 if self.start is not None else .30)
        if self.start is not None:
            if other_open and values[0] < onset[0] and stamp - self.start <= 2.5:
                return self.holding_result(eyes)
            reason = ("el derecho salió de la banda de apertura" if not other_open else
                      "perdió asimetría" if not dominant else "cierre demasiado largo"
                      if stamp - self.start > 2.5 else None)
            if reason:
                self.start = None
                return WinkResult(eyes, state="Gesto rechazado",
                                  notice=f"Izquierdo: {reason}.")
            if self.closed[0]:
                self.held = stamp - self.start
            return self.holding_result(eyes)
        if other_open and dominant and self.primed and self.closed[0]:
            self.start, self.held = stamp, 0.
            self.primed = False
            return WinkResult(eyes, holding=True, state="Izquierdo: midiendo duración",
                              prior_open_ms=self.prior_open_ms)
        if not other_open:
            self.primed = False
            notice = "Ojo derecho fuera de banda. Solo se admite el guiño izquierdo."
        elif not self.closed[0]:
            notice = "Izquierdo: señal por debajo del umbral de cierre."
        elif not dominant:
            notice = "Izquierdo: falta asimetría respecto al derecho."
        else:
            notice = "Izquierdo: falta apertura previa estable (100 ms)."
        return WinkResult(eyes, state="Gesto rechazado" if self.closed[0] or self.closed[1] else "Transición",
                          notice=notice, prior_open_ms=self.prior_open_ms)

    def holding_result(self, eyes):
        qualified = self.held >= self.duration
        state = ("Guiño unilateral válido · espera reapertura" if qualified else
                 f"Izquierdo: {self.held*1000:.0f} / {self.duration*1000:.0f} ms")
        return WinkResult(eyes, holding=True, state=state, qualified=qualified,
                          prior_open_ms=self.prior_open_ms)


@dataclass(frozen=True)
class SequenceResult:
    state: str
    button: str | None = None
    notice: str | None = None


class WinkSequence:
    def __init__(self, interval=.60, cooldown=.65):
        self.interval, self.cooldown = interval, cooldown
        self.cooldown_until = 0.
        self.reset()

    def reset(self):
        self.deadline = None
        self.second_started = False

    def update(self, stamp, gesture):
        if gesture.bilateral or gesture.interrupted:
            had_pending = self.deadline is not None
            self.reset()
            return SequenceResult("Gesto rechazado" if had_pending or gesture.bilateral else gesture.state,
                                  notice="Secuencia cancelada: " + (gesture.notice or gesture.state) if had_pending else None)
        if stamp < self.cooldown_until:
            return SequenceResult("Enfriamiento", notice="Guiño descartado durante enfriamiento."
                                  if gesture.confirmed else None)
        if self.deadline is None:
            if gesture.confirmed:
                self.deadline = stamp + self.interval
                return SequenceResult("Guiño 1 detectado · esperando segundo")
            return SequenceResult(gesture.state)
        if gesture.holding and stamp <= self.deadline:
            self.second_started = True
        if gesture.confirmed and (self.second_started or stamp <= self.deadline):
            return self.finish(stamp, "right", "Clic derecho")
        if not gesture.holding:
            self.second_started = False
        if stamp >= self.deadline and gesture.reopened:
            return self.finish(stamp, "left", "Clic izquierdo")
        return SequenceResult("Guiño 1 detectado · esperando segundo" if not self.second_started else "Guiño 2 en curso")

    def finish(self, stamp, button, state):
        self.reset()
        self.cooldown_until = stamp + self.cooldown
        return SequenceResult(state, button)
