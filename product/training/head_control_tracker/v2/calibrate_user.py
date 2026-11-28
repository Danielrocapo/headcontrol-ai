"""Calibración supervisada y uso opcional del cursor para Tracker D."""

import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

from calibration_math import GRID, HELD_OUT, Smoother, affine, compare, fit
from capture_tracker9 import CONFIRM, REPEAT, Screen, StopCapture, calibrate, primary_size
from capture_tracker import WINDOW, YELLOW
from features9 import Camera9, FEATURES, VERSION, vector
from protocol import label

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "calibration_profiles"
RUNTIME = HERE / "calibration_runtime"
GUIDED_RUNTIME = HERE / "guided_runtime"
GUIDED_V2_RUNTIME = HERE / "guided_v2_runtime"
NORMALIZER = HERE / "experiments/preliminary_20261126T044504Z_db2cc8bd/D_mlp_9_normalizador.json"
STABILITY_PX = 60.
CAPTURE_S = 2.
BLUE, GREEN = (255, 165, 40), (80, 230, 120)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Tracker:
    def __init__(self, runtime=RUNTIME):
        import onnxruntime as ort

        self.identity = read_json(runtime / "metadata.json")
        graph = runtime / self.identity["onnx"]
        norm = read_json(NORMALIZER)
        for path, key in ((graph, "onnx_sha256"), (NORMALIZER, "normalizer_sha256")):
            if sha256(path.read_bytes()).hexdigest() != self.identity[key]:
                raise ValueError(f"SHA-256 incompatible: {path.name}")
        if tuple(self.identity["features"]) != FEATURES or tuple(norm["features"]) != FEATURES or norm["fit_session"] != "TR-01":
            raise ValueError("Orden de características incompatible.")
        self.mean, self.std = np.asarray(norm["mean"]), np.asarray(norm["std"])
        if self.mean.shape != (9,) or self.std.shape != (9,) or not np.isfinite(self.mean).all() or not (
                np.isfinite(self.std) & (self.std > 0)).all():
            raise ValueError("Normalizador inválido.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])

    def predict(self, row, reference):
        features = vector(row, reference)
        tensor = ((np.array([features[k] for k in FEATURES]) - self.mean) / self.std).astype(np.float32)
        point = self.session.run(["coordinates"], {"features": tensor[None, :]})[0][0]
        return point.astype(float) if point.shape == (2,) and np.isfinite(point).all() else None


def spread(samples, scale):
    return np.ptp(np.percentile(samples, (10, 90), axis=0), axis=0) * scale


class CalibrationScreen(Screen):
    def __init__(self, camera, tracker, width, height):
        super().__init__(camera, width, height)
        self.tracker = tracker
        self.scale = np.array([width - 1, height - 1])
        self.reference = self.matrix = self.original = self.frozen = self.manual = None
        self.manual_enabled = False
        self.recent = []
        cv2.setMouseCallback(WINDOW, self.click)

    def click(self, event, x, y, flags, _):
        if self.manual_enabled and event == cv2.EVENT_LBUTTONDOWN and 0 <= x < self.width and 0 <= y < self.height:
            self.manual = np.array([x, y]) / self.scale

    def frame(self, title, instructions, trial=None, mark=True):
        if self.reference is None:
            return super().frame(title, instructions, trial, mark)
        now, row, _, _ = self.camera.read()
        self.original = self.tracker.predict(row, self.reference) if row["valid_detection"] else None
        if self.original is None:
            self.recent.clear()
        else:
            self.recent.append((now, self.original))
            self.recent = [(t, p) for t, p in self.recent if now - t <= .75]
        original = self.frozen if self.frozen is not None else self.original
        corrected = None if original is None or self.matrix is None else affine(original, self.matrix)
        target = None if trial is None else np.array([trial["target_x"], trial["target_y"]])
        self.buffer[:] = (23, 26, 30)
        entries = []
        points = ((target, YELLOW, "Objetivo"), (original, BLUE, "Original"), (corrected, GREEN, "Corregida"))
        for i, (point, color, name) in enumerate(points):
            if point is None:
                entries.append((f"{name}: --", color))
                continue
            pixels = point * self.scale
            inside = np.all((point >= 0) & (point <= 1))
            entries.append((f"{name}: X {pixels[0]:.1f}   Y {pixels[1]:.1f}" + ("" if inside else " (fuera)"), color))
            if inside:
                cv2.circle(self.buffer, tuple(np.round(pixels).astype(int)), self.radius + 3 * i, color,
                           -1 if i == 0 else 2, cv2.LINE_AA)
        if self.manual is not None:
            cv2.drawMarker(self.buffer, tuple(np.round(self.manual * self.scale).astype(int)), (246, 242, 230),
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        if target is not None:
            for point, color, name in points[1:]:
                if point is not None:
                    dx, dy = (point - target) * self.scale
                    entries.append((f"Error {name.lower()}: X {dx:+.1f}   Y {dy:+.1f} px", color))
        stable = "--"
        if len(self.recent) >= 8:
            dx, dy = spread([p for _, p in self.recent], self.scale)
            stable = (f"{'ESTABLE' if max(dx, dy) <= STABILITY_PX else 'INESTABLE'}  "
                      f"P90-P10 X {dx:.0f} / Y {dy:.0f} px (lim. {STABILITY_PX:g})")
        lines = [(title, (240, 244, 248)), *entries, ("Estabilidad: " + stable, (240, 244, 248)),
                 ("Deteccion valida" if self.original is not None else "Sin prediccion valida", (205, 213, 228)),
                 *((s, (205, 213, 228)) for s in instructions)]
        text_x = round(self.width * (.52 if target is not None and target[0] < .5 else .045))
        step = max(18, round(self.height * .031))
        font = min(.78, self.width / 2350)
        for i, (line, color) in enumerate(lines):
            cv2.putText(self.buffer, line, (text_x, step * (i + 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        font, color, 2 if i == 0 else 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, self.buffer)
        key = cv2.waitKey(1) & 0xff
        if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise StopCapture
        if cv2.getWindowImageRect(WINDOW) != (0, 0, self.width, self.height):
            raise RuntimeError("El área dibujada no coincide con el monitor principal.")
        return now, row, key


def capture_point(view, xy, phase, number, ready_hint=None):
    target = label(*xy, view.width, view.height)
    title = f"{phase.upper()} | {number}"
    hint = ready_hint or "Mira el punto amarillo. ENTER cuando estes listo; ESC para salir."
    while True:
        view.wait(title, (hint,), CONFIRM, target)
        samples, timestamps, total = [], [], 0
        start = monotonic()
        while monotonic() - start < CAPTURE_S:
            now, row, key = view.frame(title, ("Mantente estable. R para repetir.",), target)
            if key in REPEAT:
                break
            total += 1
            if view.original is not None:
                samples.append(view.original)
                timestamps.append(row["timestamp_utc"])
        else:
            if len(samples) >= 12 and len(samples) / total >= .70:
                variation = spread(samples, view.scale)
                if max(variation) <= STABILITY_PX:
                    original = np.median(samples, axis=0)
                    view.frozen = original
                    view.manual_enabled = phase == "ajuste"
                    instructions = ("Clic opcional para ajustar destino. ENTER guardar | R repetir."
                                    if view.manual_enabled else "ENTER guardar | R repetir. Conserva el objetivo amarillo.")
                    try:
                        key = view.wait(title, (instructions,), CONFIRM + REPEAT, target)
                        if key in CONFIRM:
                            manual = view.manual.tolist() if view.manual is not None else None
                            expected = [target["target_x"], target["target_y"]]
                            return dict(phase=phase, number=number, exact_target_px=[target["target_px_x"], target["target_px_y"]],
                                        original=original.tolist(), target=expected, manual=manual,
                                        correction_vector_px=((np.asarray(manual if manual is not None else expected) -
                                                               original) * view.scale).tolist(),
                                        samples=np.asarray(samples).tolist(), timestamps_utc=timestamps,
                                        valid_fraction=len(samples) / total, spread_px=variation.tolist())
                    finally:
                        view.frozen = view.manual = None
                        view.manual_enabled = False
                    continue
                hint = f"Postura inestable ({variation[0]:.0f} / {variation[1]:.0f} px). Vuelve a intentar con ENTER."
            else:
                hint = f"Detecciones insuficientes ({len(samples)}/{total}). Vuelve a intentar con ENTER."


def run_calibration(view, tracker):
    stamp = datetime.now(timezone.utc)
    folder = PROFILES / f"{stamp:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    print(f"Calibración | {folder}", flush=True)
    data = dict(date_utc=stamp.isoformat(), screen_size=[view.width, view.height],
                model=tracker.identity, feature_contract=VERSION, camera=view.camera.metadata,
                neutral=view.reference, ajuste=[], validacion=[])
    for phase, targets in (("ajuste", GRID), ("validacion", HELD_OUT)):
        for number, xy in enumerate(targets, 1):
            data[phase].append(capture_point(view, xy, phase, f"{number}/{len(targets)}"))
            atomic_json(folder / "dataset.json", data)
            if phase == "ajuste" and number >= 3:
                try:
                    view.matrix, _ = fit(data["ajuste"])
                except ValueError:
                    view.matrix = None
        if phase == "ajuste":
            view.matrix, condition = fit(data["ajuste"])
    before, after, enabled = compare(data["validacion"], view.matrix, view.scale)
    for p in (*data["ajuste"], *data["validacion"]):
        p["corrected"] = np.median(affine(p["samples"], view.matrix), axis=0).tolist()
        p["original_error_px"] = ((np.asarray(p["original"]) - p["target"]) * view.scale).tolist()
        p["corrected_error_px"] = ((np.asarray(p["corrected"]) - p["target"]) * view.scale).tolist()
    atomic_json(folder / "dataset.json", data)
    profile = dict(schema=1, date_utc=data["date_utc"], screen_size=data["screen_size"], model=data["model"],
                   feature_contract=VERSION, camera=data["camera"], matrix=view.matrix.tolist(), enabled=enabled,
                   design_condition=condition, fit_points=len(data["ajuste"]), validation_points=len(data["validacion"]),
                   metrics=dict(original=before, corrected=after))
    atomic_json(folder / "profile.json", profile)
    write_report(folder)
    print(json.dumps(profile["metrics"], indent=2), flush=True)
    print("Perfil aprobado." if enabled else "Se conserva la predicción original.", flush=True)
    if not enabled:
        view.matrix = None
    view.wait("VALIDACION COMPLETA", (f"Error medio: {before['mean_px']:.1f} -> {after['mean_px']:.1f} px",
              "Correccion aprobada." if enabled else "Prediccion original conservada.", "ESC para finalizar."), ())


def profile_matrix(path, identity, size, camera):
    profile = read_json(path)
    if profile["schema"] != 1 or profile["screen_size"] != list(size) or profile["feature_contract"] != VERSION:
        raise ValueError("Perfil incompatible con la pantalla o con el contrato facial.")
    for key in ("weights_sha256", "onnx_sha256", "normalizer_sha256"):
        if profile["model"][key] != identity[key]:
            raise ValueError("El perfil pertenece a otro modelo.")
    for key in ("source_sha256", "face_task_sha256", "camera_index", "camera_width", "camera_height", "mirrored"):
        if profile["camera"][key] != camera[key]:
            raise ValueError("El perfil pertenece a otro extractor o cámara.")
    matrix = np.asarray(profile["matrix"], dtype=float)
    if matrix.shape != (3, 2) or not np.isfinite(matrix).all():
        raise ValueError("Matriz afín inválida.")
    metrics = profile["metrics"]
    return matrix if profile["enabled"] is True and metrics["corrected"]["mean_px"] < metrics["original"]["mean_px"] else None


def write_report(folder):
    data = read_json(folder / "dataset.json")
    profile = read_json(folder / "profile.json")
    before, after = (profile["metrics"][key] for key in ("original", "corrected"))
    result = "Corrección habilitada" if profile["enabled"] else "Se conserva la predicción original"
    lines = ["# Calibración supervisada de Tracker D", "", f"**{result}.**",
             f"La decisión usa el error euclidiano medio de {after['points']} puntos reservados.", "",
             "| Métrica | Original | Afín |", "|---|---:|---:|"]
    for name, key in (("Error medio", "mean_px"), ("Mediana", "median_px"), ("P90", "p90_px"),
                      ("MAE X", "mae_x_px"), ("MAE Y", "mae_y_px")):
        lines.append(f"| {name} | {before[key]:.1f} px | {after[key]:.1f} px |")
    samples = np.concatenate([p["samples"] for p in data["validacion"]])
    frame_counts = [int(np.any((x < 0) | (x > 1), axis=1).sum()) for x in (samples, affine(samples, profile["matrix"]))]
    lines += [f"| Medianas fuera de pantalla | {before['outside']} | {after['outside']} |",
              f"| Fotogramas fuera de pantalla (de {len(samples)}) | {frame_counts[0]} | {frame_counts[1]} |", "",
              "| Objetivo en px | Error original | Error corregido | Error X/Y corregido |",
              "|---|---:|---:|---:|"]
    for point in data["validacion"]:
        original, corrected = [np.linalg.norm(point[f"{kind}_error_px"]) for kind in ("original", "corrected")]
        x, y = point["corrected_error_px"]
        tx, ty = point["exact_target_px"]
        lines.append(f"| ({tx}, {ty}) | {original:.1f} px | {corrected:.1f} px | {x:+.1f} / {y:+.1f} px |")
    gains = [np.linalg.norm(p["original_error_px"]) - np.linalg.norm(p["corrected_error_px"]) for p in data["validacion"]]
    lines += ["", f"Mejoró {sum(gain > 0 for gain in gains)} de {len(gains)} fijaciones.",
              "Las métricas usan predicciones directas, sin suavizado ni recorte. Los clics manuales se limitan al ajuste.",
              "", "[Perfil](profile.json) · [Captura y predicciones originales](dataset.json)"]
    (folder / "INFORME_CALIBRACION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def live(view, matrix, tau, allow_cursor, raw):
    smooth = Smoother(tau)
    user32 = ctypes.windll.user32
    down = armed = False
    corrected = not raw and matrix is not None
    while True:
        f8 = bool(user32.GetAsyncKeyState(0x77) & 0x8000) if allow_cursor else False
        if f8 and not down:
            armed = not armed
            smooth.reset()
        down = f8
        view.matrix = matrix if corrected else None
        status = ("Cursor activo" if armed else "Cursor pausado | F8 para activar") if allow_cursor else "Vista previa"
        now, _, key = view.frame("TRACKER D | " + ("AFIN" if corrected else "ORIGINAL"),
                                 (status, "C cambia correccion | ESC cierra."))
        if key in (ord("c"), ord("C")) and matrix is not None:
            corrected = not corrected
            smooth.reset()
            continue
        if view.original is None:
            armed = False
            smooth.reset()
            continue
        if armed:
            value = affine(view.original, view.matrix) if view.matrix is not None else view.original
            x, y = np.rint(np.clip(smooth.step(value, now), 0., 1.) * view.scale).astype(int)
            if not user32.SetCursorPos(int(x), int(y)):
                raise RuntimeError("Windows rechazó el movimiento del cursor.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--live", type=Path, metavar="PROFILE_JSON")
    modes.add_argument("--list", action="store_true")
    modes.add_argument("--report", type=Path, metavar="RUN_DIR")
    parser.add_argument("--move-cursor", action="store_true")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--smoothing-ms", type=float, default=120.)
    args = parser.parse_args()
    if (args.move_cursor or args.raw) and args.live is None:
        parser.error("--move-cursor y --raw requieren --live.")
    if not np.isfinite(args.smoothing_ms) or args.smoothing_ms < 0:
        parser.error("--smoothing-ms debe ser finito y no negativo.")
    if args.list:
        for path in sorted(PROFILES.glob("*/profile.json"), reverse=True):
            print(path)
        return
    if args.report is not None:
        write_report(args.report)
        return
    tracker = Tracker()
    width, height = primary_size()
    with ExitStack() as stack:
        camera = Camera9(stack)
        stack.callback(cv2.destroyAllWindows)
        view = CalibrationScreen(camera, tracker, width, height)
        try:
            if args.live:
                matrix = profile_matrix(args.live, tracker.identity, (width, height), camera.metadata)
                view.reference = calibrate(view)
                live(view, matrix, args.smoothing_ms / 1000., args.move_cursor, args.raw)
            else:
                view.reference = calibrate(view)
                run_calibration(view, tracker)
        except StopCapture:
            return


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
