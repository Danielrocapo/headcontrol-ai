"""Indicadores visuales de TR-02 basados en la evaluación del protocolo."""

from collections import Counter, deque
from time import monotonic

import cv2
import numpy as np

from capture_tracker import Screen, StopCapture, WINDOW, YELLOW
from features9 import FEATURES, vector
from physical_guide import diagram
from protocol import QUALITY, STABILITY_FIELDS, assess

BG = (31, 35, 43)
INK = (227, 233, 242)
MUTED = (169, 183, 201)
GOOD = (144, 229, 174)
WARN = (115, 180, 255)
ACCENT = (100, 225, 255)
NAMES = ("Yaw", "Pitch", "Roll", "Nose X", "Nose Y", "Chin X", "Chin Y", "Face center DX", "Face center DY")
AXES = ("Yaw", "Pitch", "Roll", "DX", "DY")
GUIDANCE = {
    "rotation": "Gira o inclina. Conserva el centro en la cruz.",
    "translation": "Desplaza todo el rostro. Mira al frente.",
    "combined": "Desplaza y gira con un movimiento comodo.",
}


def direction(trial):
    x, y = trial["start"]
    dx, dy = np.sign([x - .5, y - .5]).astype(int)
    words = ["IZQUIERDA" if dx < 0 else "DERECHA"] if dx else []
    if dy:
        words.append("ARRIBA" if dy < 0 else "ABAJO")
    return " / ".join(words) or "CENTRO", int(dx), int(dy)


def diagnostics(trial, rows, counts, result=None):
    cfg = QUALITY
    total = sum(counts.values())
    result = assess(trial, rows, total) if result is None else result
    reason = result["reason"]
    valid = len(rows)
    percent = 100 * valid / max(total, 1)
    details = [
        f"Detecciones: {valid}/{total} ({percent:.0f}%) | minimo {cfg['min_frames']} y {100*cfg['min_valid_fraction']:.0f}%",
    ]
    if total > valid:
        lost = sorted(((n, k) for k, n in counts.items() if k != "valid" and n), reverse=True)
        details.append("Lecturas perdidas: " + (", ".join(f"{k}={n}" for n, k in lost[:3]) or str(total - valid)))
    no_data = ("CENTRO --", "ROTACION --", "ESTABILIDAD --")
    if not rows or reason == "nonfinite_features":
        return no_data, details + (["Valores no finitos en la medicion."] if reason == "nonfinite_features" else [])
    data = np.abs(np.array([[row[key] for key in STABILITY_FIELDS] for row in rows]))
    if not np.isfinite(data).all():
        return no_data, details + ["Valores no finitos en la medicion."]
    spread = np.ptp(np.percentile([[row[k] for k in STABILITY_FIELDS] for row in rows], (10, 90), axis=0), axis=0)
    axis_p90 = np.percentile(data, 90, axis=0)
    radii = np.linalg.norm(data[:, 3:], axis=1)
    angles = np.max(data[:, :3], axis=1)
    center_fraction = float(np.mean(radii <= cfg["neutral_center"]))
    angle_fraction = float(np.mean(angles <= cfg["neutral_angle_deg"]))
    angle90, radius90 = np.percentile(angles, 90), np.percentile(radii, 90)
    limits = [cfg["stable_angle_spread_deg"]] * 3 + [cfg["stable_center_spread"]] * 2
    stable = all(v <= limit for v, limit in zip(spread, limits))
    enough = valid >= cfg["min_frames"]
    center_still = max(spread[3:]) <= cfg["stable_center_spread"]
    center_band = center_fraction >= cfg["compliance_fraction"]
    center = "CENTRO ESTABLE" if center_band and center_still else (
        "CENTRO DESPLAZADO" if not center_band else "CENTRO INESTABLE")
    rotation = "ROTACION SUFICIENTE" if angle90 >= cfg["minimum_angle_deg"] else "ROTACION INSUFICIENTE"
    stability = "POSTURA ESTABLE" if stable else "POSTURA INESTABLE"
    if not enough:
        stability = "REUNIENDO MUESTRAS"
    condition = trial["condition"]
    center_check = trial["center_control"]
    details.extend((
        f"Variacion P90-P10: yaw {spread[0]:.2f}, pitch {spread[1]:.2f}, roll {spread[2]:.2f} (lim. {limits[0]:g} grados)",
        f"Variacion centro: DX {spread[3]:.3f}, DY {spread[4]:.3f} (lim. {limits[3]:.2f})",
    ))
    if center_check:
        details.append(f"Control de deriva: radio P90 {radius90:.3f}. El centro desplazado se admite si es estable.")
    elif condition == "rotation":
        details.append(f"Centro en radio <= {cfg['neutral_center']:.2f}: {100*center_fraction:.0f}% "
                       f"(min. {100*cfg['compliance_fraction']:.0f}%)")
        details.append(f"Radio P90 {radius90:.3f} | magnitud P90 de DX {axis_p90[3]:.3f}, DY {axis_p90[4]:.3f}")
        if not center_band:
            outside = cfg["compliance_fraction"]
            axes = [name for axis, name in ((3, "horizontal"), (4, "vertical"))
                    if np.mean(data[:, axis] <= cfg["neutral_center"]) < outside]
            details.append("Desplazamiento excesivo: " + " y ".join(axes or ["radio conjunto de DX/DY"]))
    if not center_check and condition in ("rotation", "combined"):
        details.append(f"Giro P90 del max. angular: {angle90:.2f} grados (min. {cfg['minimum_angle_deg']:g})")
        if angle90 < cfg["minimum_angle_deg"]:
            _, dx, dy = direction(trial)
            axes = "Yaw insuficiente" if dx else ""
            axes += (" / " if axes else "") + ("Pitch insuficiente" if dy else "")
            details.append(f"{axes or 'Giro insuficiente'}: yaw {axis_p90[0]:.2f}, pitch {axis_p90[1]:.2f}, "
                           f"roll {axis_p90[2]:.2f} (P90 por eje, diagnostico)")
    if not center_check and condition == "translation":
        details.append(f"Angulos en banda +/-{cfg['neutral_angle_deg']:g} grados: {100*angle_fraction:.0f}% "
                       f"(min. {100*cfg['compliance_fraction']:.0f}%)")
        if angle_fraction < cfg["compliance_fraction"]:
            outside = [(float(np.mean(data[:, i] > cfg["neutral_angle_deg"])), name)
                       for i, name in enumerate(AXES[:3])]
            details.append("Fuera de banda: " + ", ".join(f"{name} {value:.0%}" for value, name in outside))
    if not center_check and condition in ("translation", "combined"):
        details.append(f"Desplazamiento radial P90: {radius90:.3f} (min. {cfg['minimum_center']:.2f})")
        if radius90 < cfg["minimum_center"]:
            details.append("Desplazamiento insuficiente para esta condicion.")
    failed = [name for name, value, limit in zip(AXES, spread, limits) if value > limit]
    if failed:
        details.append("Inestabilidad en: " + ", ".join(failed))
    return (center, rotation, stability), details


class TR02Screen(Screen):
    def __init__(self, camera, width, height):
        super().__init__(camera, width, height)
        self.panel = np.empty((844, 850, 3), dtype=np.uint8)
        self.trial = None

    def prepare_feedback(self, trial, reference, title):
        self.trial, self.reference, self.title = trial, reference, title
        self.samples = deque()
        self.last_update = -float("inf")
        self.feedback_state("PREPARACION", None)

    def feedback_state(self, phase, deadline, rows=None, counts=None, result=None):
        if getattr(self, "phase", None) != phase or result is not None:
            self.last_update = -float("inf")
        self.phase, self.deadline = phase, deadline
        self.rows, self.counts, self.result = rows, counts, result

    def finish_feedback(self):
        self.trial = None

    def frame(self, title, instructions, trial=None, mark=True):
        if self.trial is None or trial is None:
            return super().frame(title, instructions, trial, mark)
        now, row, rgb, _ = self.camera.read()
        values = vector(row, self.reference) if row["valid_detection"] else {}
        if self.rows is None:
            self.samples.append((now, dict(row, **values)))
            while self.samples and now - self.samples[0][0] > .85:
                self.samples.popleft()
        if now - self.last_update >= .12:
            recent = [r for _, r in self.samples]
            rows = self.rows if self.rows is not None else [r for r in recent if r["valid_detection"]]
            counts = self.counts if self.counts is not None else Counter(r["status"] for r in recent)
            result = assess(self.trial, rows, sum(counts.values())) if self.result is None else self.result
            self.indicators, self.details = diagnostics(self.trial, rows, counts, result)
            self.reason = result["reason"]
            self.has_losses = sum(counts.values()) > len(rows)
            self.last_update = now
        self.paint(row, rgb, values, instructions, trial)
        cv2.imshow(WINDOW, self.buffer)
        key = cv2.waitKey(1) & 0xff
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise StopCapture
        bounds = cv2.getWindowImageRect(WINDOW)
        if bounds != (0, 0, self.width, self.height):
            raise RuntimeError(f"Área de dibujo {bounds}; se requiere (0, 0, {self.width}, {self.height}).")
        return now, row, key

    def paint(self, row, rgb, values, instructions, target):
        panel = self.panel
        panel[:] = BG

        def text(line, x, y, scale=.6, color=INK, weight=1):
            cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

        condition = self.trial["condition"]
        center = self.trial["center_control"]
        requested, dx, dy = direction(self.trial)
        text(self.title, 22, 37, .87, INK, 2)
        text("Objetivo: " + requested, 22, 69, .7, ACCENT, 2)
        remaining = ("en espera" if self.result is None else "0.0 s") if self.deadline is None else (
            f"{max(0., self.deadline - monotonic()):.1f} s")
        text(f"{self.phase}  |  Tiempo restante: {remaining}", 22, 100, .65, INK, 2)
        if rgb is not None:
            preview = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (420, 236))
            cx, cy = self.reference["center_px"]
            scale_x, scale_y = 420 / row["camera_width"], 236 / row["camera_height"]
            neutral = (round(cx * scale_x), round(cy * scale_y))
            if row["valid_detection"]:
                actual = (round(row["face_center_x_px"] * scale_x), round(row["face_center_y_px"] * scale_y))
                cv2.line(preview, neutral, actual, ACCENT, 2, cv2.LINE_AA)
                cv2.circle(preview, actual, 5, ACCENT, -1, cv2.LINE_AA)
            cv2.drawMarker(preview, neutral, INK, cv2.MARKER_CROSS, 21, 2, cv2.LINE_AA)
            panel[117:353, 22:442] = preview
        kind = "neutral" if center else condition if condition == "combined" else f"{condition}_{'x' if dx else 'y'}"
        sign = dx or dy
        guide = diagram(kind, "TR1" if sign < 0 else "TR2")
        panel[117:252, 465:820] = cv2.resize(guide[:205], (355, 135))
        if dx or dy:
            cv2.arrowedLine(panel, (637, 284), (637 + dx * 41, 284 + dy * 24), ACCENT, 3, cv2.LINE_AA, tipLength=.24)
        instruction = ("Vuelve a tu centro y permanece quieto." if center else
                       "Gira e inclina. Conserva el centro en la cruz." if condition == "rotation" and dx and dy
                       else GUIDANCE[condition])
        text(instruction, 465, 335, .43, MUTED)
        text("Cruz: neutral    Punto y linea: centro actual", 22, 376, .51, MUTED)
        status = ("DETECCION VALIDA" if row["valid_detection"] else
                  "ROSTRO NO DETECTADO" if not row["face_detected"] else "ROSTRO SIN LECTURA VALIDA")
        text(status, 465, 376, .53, GOOD if row["valid_detection"] else WARN, 2)
        for i, (key, name) in enumerate(zip(FEATURES, NAMES)):
            unit = " deg" if i < 3 else ""
            shown = f"{values[key]:+.3f}{unit}" if key in values else "--"
            text(f"{name}  {shown}", 22 + i % 3 * 280, 412 + i // 3 * 29)
        indicators = list(self.indicators)
        if center:
            indicators[1] = "GIRO NO EXIGIDO"
        elif condition == "translation":
            # La banda angular de traslación ya se detalla con sus porcentajes.
            indicators[1] = "VIGILA BANDA DE GIRO"
        for x, indicator in zip((22, 303, 592), indicators):
            color = WARN if any(s in indicator for s in ("DESPLAZADO", "INSUFICIENTE", "INESTABLE")) else (
                GOOD if indicator in ("CENTRO ESTABLE", "ROTACION SUFICIENTE", "POSTURA ESTABLE") else MUTED)
            text(indicator, x, 510, .53, color, 2)
        valid_window = self.reason == "ok" and (row["valid_detection"] or self.result is not None)
        captions = {
            "insufficient_detection": ("PERDIDA DE DETECCION / MUESTRAS INSUFICIENTES" if self.has_losses
                                       else "MUESTRAS INSUFICIENTES"),
            "unstable_fixation": "POSTURA INESTABLE DURANTE LA MEDICION",
            "rotation_mismatch": "CENTRO O GIRO FUERA DEL CRITERIO DE ROTACION",
            "translation_mismatch": "DESPLAZAMIENTO O GIRO FUERA DEL CRITERIO DE TRASLACION",
            "combined_mismatch": "FALTA GIRO O DESPLAZAMIENTO",
            "nonfinite_features": "CARACTERISTICAS NO VALIDAS",
        }
        caption = "CONDICION CUMPLIDA" if valid_window else captions.get(self.reason, "CONDICION PENDIENTE")
        text(caption, 22, 546, .62, GOOD if valid_window else WARN, 2)
        window_name = ("Resultados de la fijacion" if self.result is not None else
                       "Medicion acumulada" if self.rows is not None else "Guia en vivo: ultimas lecturas")
        text(window_name, 555, 565, .42, MUTED)
        for i, line in enumerate(self.details):
            text(line, 22, 578 + i * 21, .50)
        text(instructions[-1], 22, 816, .56, ACCENT)
        if self.result is not None:
            text(instructions[0], 22, 790, .56, INK)
        self.buffer[:] = (23, 26, 30)
        scale = min(.46 * self.width / panel.shape[1], .94 * self.height / panel.shape[0])
        w, h = round(panel.shape[1] * scale), round(panel.shape[0] * scale)
        x = round(self.width * .02) if target["target_x"] >= .5 else self.width - w - round(self.width * .02)
        y = (self.height - h) // 2
        self.buffer[y:y+h, x:x+w] = cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)
        cv2.circle(self.buffer, (target["target_px_x"], target["target_px_y"]), self.radius, YELLOW, -1, cv2.LINE_AA)
