"""Lógica de movimiento y gestos, independiente de la cámara y la interfaz."""
from collections import deque
from dataclasses import dataclass
from math import hypot


@dataclass
class PoseSample:
    x: float
    y: float


def relative_pose(nose, eye_left, eye_right):
    """Orientación facial aproximada, normalizada por distancia entre ojos."""
    middle_x = (eye_left[0] + eye_right[0]) / 2
    middle_y = (eye_left[1] + eye_right[1]) / 2
    distance = max(hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]), 0.001)
    return PoseSample((nose[0] - middle_x) / distance, (nose[1] - middle_y) / distance)


class HeadController:
    """Convierte la desviación respecto al centro calibrado en velocidad del cursor."""

    def __init__(self):
        self.neutral = None
        self.samples = deque(maxlen=30)
        self.calibrating = True
        self.filtered_x = 0.0
        self.filtered_y = 0.0
        self.remainder_x = 0.0
        self.remainder_y = 0.0
        self.sensitivity = 1.0
        self.max_speed = 900.0
        self.reverse_x = True
        self.reverse_y = False

    def calibrate(self):
        self.samples.clear()
        self.neutral = None
        self.calibrating = True
        self.filtered_x = self.filtered_y = 0.0
        self.remainder_x = self.remainder_y = 0.0

    def observe(self, pose):
        """Se calibra promediando 30 lecturas con la cabeza en posición neutral."""
        if pose is None:
            self.filtered_x = self.filtered_y = 0.0
            self.remainder_x = self.remainder_y = 0.0
            return False
        if self.calibrating:
            self.samples.append(pose)
            if len(self.samples) == self.samples.maxlen:
                self.neutral = PoseSample(
                    sum(p.x for p in self.samples) / len(self.samples),
                    sum(p.y for p in self.samples) / len(self.samples),
                )
                self.calibrating = False
            return False
        return True

    @property
    def progress(self):
        return len(self.samples) / self.samples.maxlen

    @staticmethod
    def _axis(delta, deadzone, full_speed_delta):
        extent = max(0.0, abs(delta) - deadzone)
        if extent == 0:
            return 0.0
        factor = min(1.0, extent / full_speed_delta)
        return (1 if delta > 0 else -1) * factor ** 1.35

    def move(self, pose, dt, scrolling=False):
        """Devuelve dx/dy en píxeles, o pasos de scroll vertical en modo scroll."""
        if self.neutral is None or pose is None:
            return (0, 0, 0)
        dt = min(max(dt, 0.0), 0.12)
        raw_x = (pose.x - self.neutral.x) * self.sensitivity
        raw_y = (pose.y - self.neutral.y) * self.sensitivity
        self.filtered_x = self.filtered_x * 0.68 + raw_x * 0.32
        self.filtered_y = self.filtered_y * 0.68 + raw_y * 0.32
        horizontal = self._axis(self.filtered_x, 0.055, 0.26)
        vertical = self._axis(self.filtered_y, 0.075, 0.27)
        if scrolling:
            self.remainder_y += vertical * dt * 18
            steps = int(self.remainder_y)
            self.remainder_y -= steps
            return (0, 0, -steps if not self.reverse_y else steps)
        vx = -horizontal if self.reverse_x else horizontal
        vy = -vertical if self.reverse_y else vertical
        self.remainder_x += vx * self.max_speed * dt
        self.remainder_y += vy * self.max_speed * dt
        dx, dy = int(self.remainder_x), int(self.remainder_y)
        self.remainder_x -= dx
        self.remainder_y -= dy
        return (dx, dy, 0)


class WinkDetector:
    """Requiere un solo ojo cerrado intencionalmente y luego soltar el guiño."""

    def __init__(self, threshold=0.7, other_max=0.38, min_hold=0.17, cooldown=0.95):
        self.threshold = threshold
        self.other_max = other_max
        self.min_hold = min_hold
        self.cooldown = cooldown
        self.current = None
        self.started = 0.0
        self.last_click = float("-inf")

    def reset(self):
        self.current = None
        self.started = 0.0

    def update(self, left, right, now):
        candidate = None
        if left >= self.threshold and right < self.other_max:
            candidate = "left"
        elif right >= self.threshold and left < self.other_max:
            candidate = "right"
        if candidate is not None:
            if candidate != self.current:
                self.current, self.started = candidate, now
            return None
        if left >= self.threshold or right >= self.threshold:
            self.reset()
            return None
        click = None
        if self.current and self.min_hold <= now - self.started <= 1.1:
            if now - self.last_click >= self.cooldown:
                click = self.current
                self.last_click = now
        self.reset()
        return click


class MouthScrollToggle:
    """Abrir la boca deliberadamente y mantenerla permite alternar el scroll."""

    def __init__(self):
        self.started = None
        self.armed = True
        self.last_toggle = float("-inf")
        self.scroll = False

    def update(self, jaw_open, now):
        if jaw_open > 0.73:
            if self.started is None:
                self.started = now
            if self.armed and now - self.started >= 1.0 and now - self.last_toggle >= 1.6:
                self.scroll = not self.scroll
                self.last_toggle = now
                self.armed = False
                return True
        else:
            self.started = None
            self.armed = True
        return False
