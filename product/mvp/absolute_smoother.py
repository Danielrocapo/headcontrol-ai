"""Suavizado temporal de coordenadas absolutas, con umbrales en píxeles."""

from collections import deque

import numpy as np


PROFILES = {"Suave": (.22, .035), "Medio": (.35, .055)}


class AdaptiveSmoother:
    def __init__(self, scale, profile, deadzone_px):
        self.scale = np.asarray(scale)
        self.rest, self.moving = PROFILES[profile]
        self.deadzone_px = deadzone_px
        self.reset()

    def reset(self):
        self.value = self.time = self.previous_target = None
        self.velocity = np.zeros(2)

    def step(self, target, now):
        if self.value is None:
            self.value = np.array(target, dtype=float, copy=True)
        dt = max(now - self.time, 0.) if self.time is not None else 0.
        delta = target - self.value
        if dt > 0:
            if self.previous_target is not None:
                speed = (target - self.previous_target) * self.scale / dt
                self.velocity += (-np.expm1(-dt / .08)) * (speed - self.velocity)
            distance = np.linalg.norm(delta * self.scale)
            intent = np.clip(max((distance - 16.) / 58., np.linalg.norm(self.velocity) / 380.), 0., 1.)
            weight = intent**2 * (3. - 2. * intent)
            tau = self.rest + (self.moving - self.rest) * weight
            goal = np.rint(np.clip(target, 0., 1.) * self.scale)
            quiet = (np.abs(delta * self.scale) <= self.deadzone_px) & (goal > 0) & (goal < self.scale)
            self.value += -np.expm1(-dt / tau) * np.where(quiet, 0., delta)
        self.previous_target = np.array(target, copy=True)
        self.time = now
        return self.value


class SteadyAdaptiveSmoother(AdaptiveSmoother):
    """Extiende el filtro anterior con confirmación de saltos y retención en reposo."""

    def __init__(self, scale, profile, deadzone_px, stability_px):
        self.stability_px = stability_px
        super().__init__(scale, profile, deadzone_px)

    def reset(self):
        super().reset()
        self.suspect = self.quiet_since = self.exit_vector = None
        self.holding = False

    def step(self, target, now):
        if self.previous_target is not None and self.is_outlier(target, now):
            return self.value
        before = self.value.copy() if self.value is not None else None
        value = super().step(target, now)
        if before is not None and self.stability_px > 0:
            self.retain(target, before, now)
        return value

    def is_outlier(self, target, now):
        # No extrapolar la velocidad por más de dos fotogramas típicos.
        dt = np.clip(now - self.time, 0., .07)
        displacement = (target - self.previous_target) * self.scale
        expected = self.velocity * dt
        limit = 23. + .70 * np.linalg.norm(expected)
        if np.linalg.norm(displacement) <= 23. or np.linalg.norm(displacement - expected) <= limit:
            self.suspect = None
            return False
        if self.suspect is not None:
            jump = (self.suspect - self.previous_target) * self.scale
            length = np.linalg.norm(jump)
            if length > 0:
                progress = np.dot(displacement, jump) / length
                sideways = np.linalg.norm(displacement - progress * jump / length)
                if progress >= .55 * length and sideways <= max(14., .35 * length):
                    if self.holding:
                        self.exit_vector = (self.suspect - self.value) * self.scale
                    self.suspect = None
                    return False
        self.suspect = np.array(target, copy=True)
        return True

    def retain(self, target, before, now):
        radius = self.stability_px
        goal = np.rint(np.clip(target, 0., 1.) * self.scale)
        edge = np.any(((goal == 0) | (goal == self.scale)) &
                      (goal != np.rint(np.clip(before, 0., 1.) * self.scale)))
        if edge:
            self.holding, self.quiet_since, self.exit_vector = False, None, None
            return
        displacement = (target - before) * self.scale
        distance = np.linalg.norm(displacement)
        if self.holding:
            outward_speed = np.dot(displacement, self.velocity) / distance if distance > 0 else 0.
            departure = distance > 1.65 * radius or (distance > radius and outward_speed >= 18.)
            if departure and self.exit_vector is not None:
                coherent = np.dot(displacement, self.exit_vector) >= .6 * distance * np.linalg.norm(self.exit_vector)
                if coherent:
                    self.holding, self.quiet_since, self.exit_vector = False, None, None
                    return
            self.exit_vector = displacement if departure else None
            self.value[:] = before
            return
        quiet = distance <= 1.5 * radius and np.linalg.norm((self.value - before) * self.scale) <= 1.2
        if quiet:
            if self.quiet_since is None:
                self.quiet_since = now
            self.holding = now - self.quiet_since >= .12
        else:
            self.quiet_since = None


class StableFixation:
    """Ancla una coordenada absoluta y usa la tendencia reciente para liberarla."""

    quiet_spread_px = 8.0
    close_px = 5.0

    def __init__(self, exit_px=20., confirm_s=.06):
        self.exit_px, self.confirm_s = exit_px, confirm_s
        self.samples = deque()
        self.reset()

    def reset(self):
        self.anchor = self.raw_anchor = self.pose_anchor = None
        self.interrupt()

    def interrupt(self):
        """Descarta evidencia temporal durante el bloqueo ocular."""
        self.samples.clear()
        self.departure_since = self.departure_direction = None
        self.state = "ANCLADO" if self.anchor is not None else "MOVIENDO"

    def step(self, stamp, raw, pose, current, scale, ready):
        """Devuelve True si acaba de liberar el ancla."""
        if self.samples and stamp - self.samples[-1][0] > .12:
            self.interrupt()
        point = np.clip(raw, 0., 1.) * scale
        self.samples.append((stamp, point, pose))
        while len(self.samples) >= 2 and stamp - self.samples[1][0] >= .20:
            self.samples.popleft()
        if self.anchor is not None:
            if self.should_release(stamp, scale):
                self.reset()
                return True
        elif ready:
            self.consider_anchor(stamp, current, scale)
        return False

    def consider_anchor(self, stamp, current, scale):
        times, predictions, poses = zip(*self.samples)
        raw = np.array(predictions)
        middle = np.median(raw, axis=0)
        rounded = np.rint(middle)
        waiting_for_edge = np.any(((rounded == 0) | (rounded == scale)) & (current != rounded))
        close_to_pointer = np.linalg.norm(middle - current) <= self.close_px
        spread = np.percentile(np.linalg.norm(raw - middle, axis=1), 80)
        n = max(1, len(raw)//3)
        elapsed = stamp - times[0]
        speed = np.linalg.norm(np.median(raw[-n:], axis=0) - np.median(raw[:n], axis=0)) / max(elapsed, .033)
        stable = not waiting_for_edge and spread <= self.quiet_spread_px and speed <= 32. and close_to_pointer
        self.pose_anchor = None
        if all(p is not None for p in poses):
            angles = np.array(poses)
            self.pose_anchor = np.median(angles, axis=0)
            pose_spread = np.percentile(np.linalg.norm(angles - self.pose_anchor, axis=1), 80)
            stable &= pose_spread <= .65
        self.state = "ESTABILIZANDO" if stable else "MOVIENDO"
        if stable and elapsed >= .199 and len(raw) >= 5:
            self.raw_anchor = middle
            self.anchor = current.copy()
            self.state = "ANCLADO"

    def should_release(self, stamp, scale):
        if len(self.samples) < 5:
            return False
        history = list(self.samples)
        first_time, _, _ = history[1]
        last_time, _, _ = history[-2]
        past = np.median([row[1] for row in history[:3]], axis=0)
        latest = np.median([row[1] for row in history[-3:]], axis=0)
        direction = latest - past
        speed = np.linalg.norm(direction) / max(last_time - first_time, .033)
        distance = np.linalg.norm(latest - self.raw_anchor)
        past_poses, recent_poses = [row[2] for row in history[:3]], [row[2] for row in history[-3:]]
        have_pose = self.pose_anchor is not None and all(p is not None for p in past_poses + recent_poses)
        pose_moves = False
        if have_pose:
            recent_pose = np.median(recent_poses, axis=0)
            pose_moves = (np.linalg.norm(recent_pose - np.median(past_poses, axis=0)) >= .15
                          and np.linalg.norm(recent_pose - self.pose_anchor) >= .20)
        goal = np.rint(latest)
        edge_diff = np.any(((goal == 0) | (goal == scale)) & (goal != self.anchor))
        small_exit = min(7., self.exit_px * .4)
        fast = distance >= self.exit_px and speed >= (55. if pose_moves else 95.)
        precise = distance >= small_exit and speed >= (24. if pose_moves else 42.) and (pose_moves or not have_pose)
        finishing_edge = edge_diff and pose_moves and distance >= .5 and speed >= 5.
        if not (fast or precise or finishing_edge):
            self.departure_since = self.departure_direction = None
            return False
        aligned = self.departure_direction is not None and np.dot(direction, self.departure_direction) >= (
            .65 * np.linalg.norm(direction) * np.linalg.norm(self.departure_direction))
        if not aligned:
            self.departure_since, self.departure_direction = stamp, direction
        return stamp - self.departure_since >= self.confirm_s


class ConsensusFixation(StableFixation):
    """Confirma la dirección de ONNX y de la pose ya medida por el tracker."""

    quiet_spread_px = 2.6
    close_px = 3.0
    fine_px = 3.0

    def interrupt(self):
        super().interrupt()
        self.departure_pose = None
        self.clear_weak()

    def clear_weak(self):
        self.weak_since = self.weak_pose = None
        self.weak_count = 0

    def step(self, stamp, raw, pose, current, scale, ready):
        self.release_target = self.fine_direction = None
        return super().step(stamp, raw, pose, current, scale, ready)

    def consider_anchor(self, stamp, current, scale):
        # The experimental release requires a facial reference.
        if all(pose is not None for _, _, pose in self.samples):
            super().consider_anchor(stamp, current, scale)

    @staticmethod
    def aligned(current, reference):
        lengths = np.linalg.norm(current) * np.linalg.norm(reference)
        return lengths > 0 and np.dot(current, reference) >= .65 * lengths

    def should_release(self, stamp, scale):
        recent = list(self.samples)[-3:]
        if len(recent) < 3 or any(pose is None for _, _, pose in recent):
            return False
        point = np.median([row[1] for row in recent], axis=0)
        pose = np.median([row[2] for row in recent], axis=0)
        direction = point - self.raw_anchor
        face_direction = pose - self.pose_anchor
        distance = np.linalg.norm(direction)
        facial = np.linalg.norm(face_direction)
        goal = np.rint(point)
        edge = np.any(((goal == 0) | (goal == scale)) & (goal != self.anchor))
        edge = edge and distance >= .5 and facial >= .30
        if not edge and (distance < self.fine_px - 1e-8 or facial < .28):
            self.departure_since = self.departure_direction = self.departure_pose = None
            self.clear_weak()
            return False

        if self.departure_since is None or not (
                self.aligned(direction, self.departure_direction)
                and self.aligned(face_direction, self.departure_pose)):
            self.departure_since, self.departure_direction, self.departure_pose = stamp, direction, face_direction
            self.clear_weak()

        large = distance >= self.exit_px
        if large and not edge and facial < .70:
            if self.weak_since is None:
                self.weak_since, self.weak_pose = stamp, pose
            self.weak_count += 1
            confirmed = (self.weak_count >= 4 and stamp - self.weak_since >= max(.13, self.confirm_s)
                         and np.linalg.norm(pose - self.weak_pose) >= .13)
            if confirmed:
                self.release_target = point / scale
            return confirmed

        self.clear_weak()
        if stamp - self.departure_since < self.confirm_s:
            return False
        self.release_target = point / scale
        if not large:
            self.fine_direction = direction
        return True


class ConsensusSmoother(SteadyAdaptiveSmoother):
    """Handoff breve para una salida fina ya confirmada, bajo los límites de Safety."""

    def reset(self):
        super().reset()
        self.fine_direction = None

    def step(self, target, now):
        if self.fine_direction is not None:
            delta = (np.clip(target, 0., 1.) - self.value) * self.scale
            distance = np.linalg.norm(delta)
            if distance <= 10. and (distance <= 1. or np.dot(delta, self.fine_direction) > 0.):
                self.value += delta * min(1., 2. / max(distance, 1.)) / self.scale
                self.previous_target = np.array(target, copy=True)
                self.time = now
                if distance <= 2.:
                    self.fine_direction = None
                return self.value
            self.fine_direction = None
        return super().step(target, now)
