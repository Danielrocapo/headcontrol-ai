"""Captura HeadControlNet Tracker v2. La salida es exclusivamente visual."""

import argparse
from collections import Counter
from contextlib import ExitStack
import math
from pathlib import Path
import sys
from time import monotonic
from uuid import uuid4

import cv2

from features9 import (
    ANGULAR_SPREAD_MAX, CENTER_SPREAD_MAX, SPAN_SPREAD_MAX, Camera9, FEATURES,
    VERSION as FEATURE_VERSION, neutral_reference, vector,
)
from protocol import (
    FIXATION_S, QUALITY, SETTLE_S, SPLITS, TR01_PROTOCOL, TR02_PROTOCOL, TR03_PROTOCOL, TRAJECTORY_S, VERSION,
    assess, label, plan, target_for,
)
from session9 import FRAME_FIELDS, Session9
from storage import dumps, write_csv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from capture_tracker import Screen, StopCapture, primary_size
from feedback_tr02 import TR02Screen

REPEAT = (ord("r"), ord("R"))
CONFIRM = (10, 13)
CAPTURE_PROTOCOLS = {"TR-01": TR01_PROTOCOL, "TR-02": TR02_PROTOCOL, "TR-03": TR03_PROTOCOL}
INSTRUCTIONS = {
    "natural": ("MOVIMIENTO NATURAL", "Apunta con la cabeza al punto con un movimiento comodo."),
    "translation": ("TRASLACION", "Desplaza el rostro manteniendo la orientacion al frente."),
    "rotation": ("ROTACION", "Gira o inclina la cabeza manteniendo el centro facial cerca de neutral."),
    "combined": ("COMBINADO", "Usa una combinacion natural, sin forzar el movimiento."),
}
QUALITY_HELP = {
    "insufficient_detection": "Deteccion insuficiente. Revisa la posicion y la luz.",
    "nonfinite_features": "La deteccion contiene valores invalidos.",
    "trajectory_incomplete": "Faltan detecciones a lo largo de la trayectoria.",
    "unstable_fixation": "Demasiado movimiento durante la fijacion.",
    "center_not_neutral": "El control central debe estar cerca de neutral.",
    "translation_mismatch": "Mueve el centro del rostro y procura mantener la orientacion.",
    "rotation_mismatch": "Conserva el centro del rostro y usa giro o inclinacion.",
    "combined_mismatch": "Hace falta algo de giro y desplazamiento. Muevete con comodidad.",
}


def calibrate(view):
    feedback = "Mira al frente, relaja la cabeza y deja el rostro quieto."
    while True:
        view.wait("CALIBRACION NEUTRAL", (feedback, "ESPACIO para medir | ESC para salir."), (32,))
        start = monotonic()
        rows = []
        while monotonic() - start < 2.5:
            now, row, _ = view.frame("MIDIENDO NEUTRAL", ("Mantente en una posicion comoda.",))
            if now - start >= .5:
                rows.append(row)
        try:
            return neutral_reference(rows)
        except ValueError:
            feedback = "Referencia inestable o insuficiente. Vuelve al centro y repite."


def session_reference(view, store):
    retained = store.reference()
    camera = view.camera.metadata
    candidate = calibrate(view)
    if retained is None:
        key = store.baseline(candidate["raw"], camera, {k: v for k, v in candidate.items() if k != "raw"})
        return key, candidate
    key, baseline, metadata = retained
    if metadata != camera:
        raise RuntimeError("Cambió la cámara o el extractor de la sesión. Restaura el entorno original.")
    span = baseline["interocular_px"]
    angle = [abs((a - b + 180) % 360 - 180) for a, b in zip(candidate["raw"][:3], baseline["raw"][:3])]
    offset = math.dist(candidate["center_px"], baseline["center_px"]) / span
    scale_change = abs(candidate["interocular_px"] - span) / span
    if (max(angle) > QUALITY["neutral_angle_deg"] or offset > QUALITY["neutral_center"] or
            scale_change > SPAN_SPREAD_MAX):
        if store.config["session_id"] == "TR-02":
            yaw, pitch, roll = angle
            raise RuntimeError(
                f"Neutral incompatible al reanudar. Yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}° "
                f"(máx. {QUALITY['neutral_angle_deg']:g}° por eje); centro={offset:.3f} "
                f"(máx. {QUALITY['neutral_center']:.2f}); escala={scale_change:.1%} "
                f"(máx. {SPAN_SPREAD_MAX:.0%}). Restaura la posición y reabre la sesión."
            )
        raise RuntimeError("El neutral no coincide con el de la sesión. Restaura la posición y vuelve a abrirla.")
    return key, baseline


def run_trial(view, store, trial, ref, ref_id, number):
    name, instruction = INSTRUCTIONS[trial["condition"]]
    title = f'{number} | {name} | {trial["target_id"]}'
    moving = trial["task"] == "trajectory"
    natural = trial["condition"] == "natural"
    center_check = trial.get("center_drift_check", False)

    def target(progress):
        return target_for(trial, progress, view.width, view.height)

    start_target = target(0)
    if moving and natural:
        instruction = "Sigue el punto con un movimiento natural de cabeza."
    if trial["center_control"]:
        instruction = ("Control central. Vuelve a tu centro comodo y permanece en reposo." if natural or center_check
                       else "Control central. Mantente en tu postura neutral.")
    wait_ready = natural or store.config["session_id"] == "TR-02"
    feedback = view if isinstance(view, TR02Screen) else None
    if feedback:
        feedback.prepare_feedback(trial, ref, title)
    start_hint = ("ENTER cuando estes en la postura final | ESC para salir." if wait_ready
                  else "ESPACIO para iniciar | ESC para salir.")
    preparation = ("Ponte en el punto inicial antes de que empiece a moverse." if moving else
                   "Coloca la cabeza para el punto amarillo.")
    view.wait(title, (instruction, preparation,
                      start_hint), CONFIRM if wait_ready else (32,), start_target)
    rows, counts = [], Counter()
    phase = "ESTABILIZANDO | R para repetir | ESC para salir."
    begun, previous = monotonic(), 0.
    try:
        if feedback:
            feedback.feedback_state("ESTABILIZANDO", begun + SETTLE_S)
        while monotonic() - begun < SETTLE_S:
            _, _, key = view.frame(title, (instruction, phase), start_target)
            if key in REPEAT:
                store.attempt(trial, ref_id, "repeated", rows, dict(reason="operator_repeat", counts={}))
                return
            previous = monotonic()
        begun = monotonic()
        duration = TRAJECTORY_S if moving else FIXATION_S
        progress = 0.
        shown = start_target
        if feedback:
            feedback.feedback_state("MEDICION EN CURSO", begun + duration, rows, counts)
        while monotonic() - begun < duration:
            phase = "MIDIENDO | Sigue el punto." if moving else "MIDIENDO | Manten la postura."
            next_progress = min(1., (monotonic() - begun) / duration) if moving else 0.
            upcoming = target(next_progress)
            now, row, key = view.frame(title, (instruction, phase, "R para repetir | ESC para salir."), upcoming)
            if key in REPEAT:
                store.attempt(trial, ref_id, "repeated", rows, dict(reason="operator_repeat", counts=dict(counts)))
                return
            if now - begun < duration:
                counts[row["status"]] += 1
                if row["valid_detection"]:
                    rows.append(dict(row, **vector(row, ref), **shown, target_progress=progress,
                                     frame_monotonic_s=now, target_presented_monotonic_s=previous))
            # Screen.frame lee la cámara antes de presentar el punto siguiente.
            shown, progress, previous = upcoming, next_progress, monotonic()
        quality = dict(assess(trial, rows, sum(counts.values())), counts=dict(counts))
        good = quality["reason"] == "ok"
        decision = ("ENTER si realizaste la tarea indicada | R para repetir." if good
                    else QUALITY_HELP[quality["reason"]] + " Pulsa R.")
        if feedback:
            feedback.feedback_state("ACEPTABLE" if good else "RECHAZADA", None, rows, counts, quality)
            decision = ("ENTER para confirmar | R para repetir." if good
                        else "Revisa las medidas y pulsa R para repetir.")
        key = view.wait("MEDICION TERMINADA", (decision, "ESC para guardar y salir."),
                        CONFIRM + REPEAT if good else REPEAT, target(float(moving)))
        status = "accepted" if key in CONFIRM else ("repeated" if good else "quality_rejected")
    except StopCapture:
        store.attempt(trial, ref_id, "cancelled", rows, dict(reason="operator_exit", counts=dict(counts)))
        raise
    finally:
        if feedback:
            feedback.finish_feedback()
    store.attempt(trial, ref_id, status, rows, quality)


def capture_session(store):
    if not store.pending():
        return
    width, height = store.config["screen_size"]
    if primary_size() != (width, height):
        raise RuntimeError("La resolución del monitor principal cambió para esta sesión.")
    with ExitStack() as stack:
        camera = Camera9(stack)
        stack.callback(cv2.destroyAllWindows)
        screen = TR02Screen if store.config["session_id"] == "TR-02" else Screen
        view = screen(camera, width, height)
        try:
            ref_id, ref = session_reference(view, store)
            plan_size = len(store.config["plan"])
            while pending := store.pending():
                run_trial(view, store, pending[0], ref, ref_id, f"{plan_size - len(pending) + 1}/{plan_size}")
            view.wait("SESION COMPLETA", ("Presiona ESC para cerrar.",), (), mark=False)
        except StopCapture:
            return


def smoke():
    folder = HERE / "technical" / uuid4().hex[:12]
    width, height = primary_size()
    target = label(.50, .50, width, height)
    folder.mkdir(parents=True)
    frames, ref = [], None
    error = None
    with ExitStack() as stack:
        camera = Camera9(stack)
        stack.callback(cv2.destroyAllWindows)
        view = Screen(camera, width, height)
        start = monotonic()
        try:
            # Presenta el objetivo primero y luego asocia las lecturas con su aparición.
            view.frame("PRUEBA TECNICA", ("Mira al frente. El punto permanece en el centro.",), target)
            presented = monotonic()
            while monotonic() - start < 10:
                now, row, _ = view.frame("PRUEBA TECNICA", ("Mira al frente con la cabeza comoda.",), target)
                frames.append((now, row))
        except StopCapture:
            error = "Prueba técnica cancelada."
    for start_at in (0.5, 2.5, 4.5, 6.5):
        try:
            subset = [r for t, r in frames if start_at <= t - start < start_at + 2.0]
            ref = neutral_reference(subset)
            break
        except ValueError:
            continue
    if ref is None:
        error = error or "La cámara no produjo una referencia neutral estable."
    rows = []
    for now, row in frames:
        values = vector(row, ref) if ref is not None and row["valid_detection"] else {}
        rows.append(dict(row, **values, **target, frame_monotonic_s=now, target_progress=0.,
                         target_presented_monotonic_s=presented))
    write_csv(folder / "fotogramas_tecnicos.csv", ("screen_width", "screen_height", *FRAME_FIELDS),
              (dict(r, screen_width=width, screen_height=height) for r in rows))
    result = dict(split="technical", camera=camera.metadata, screen_size=[width, height], neutral=ref,
                  detections=dict(Counter(r["status"] for r in rows)), error=error)
    (folder / "metadata.json").write_text(dumps(result) + "\n", encoding="utf-8")
    print(f'{len(rows)} fotogramas | {result["detections"]} | {folder}')
    if error:
        print(error)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--session", choices=SPLITS)
    mode.add_argument("--export", choices=SPLITS, help="Exporta una sesión existente sin abrir la cámara.")
    mode.add_argument("--smoke", action="store_true", help="Prueba de diez segundos, sin crear dataset.")
    mode.add_argument("--physical-check", action="store_true", help="PHYSICAL_CHECK guiado, separado de las sesiones.")
    parser.add_argument("--distance-cm", type=float)
    args = parser.parse_args()
    if args.smoke:
        return smoke()
    if args.physical_check:
        from physical_check import run_physical_check
        return run_physical_check()
    session = args.session or args.export
    folder = HERE / "sessions" / session
    config = None
    if args.session and not (folder / "session.sqlite3").is_file():
        if args.distance_cm is None or not math.isfinite(args.distance_cm) or args.distance_cm <= 0:
            parser.error("Una sesión nueva requiere una distancia positiva en centímetros.")
        config = dict(session_id=session, split=SPLITS[session], distance_cm=args.distance_cm,
                      screen_size=primary_size(), plan=plan(session), contract=VERSION,
                      feature_contract=FEATURE_VERSION, features=FEATURES, quality=QUALITY,
                      neutral_thresholds=dict(angle_spread_deg=ANGULAR_SPREAD_MAX, center_spread=CENTER_SPREAD_MAX,
                                              span_spread=SPAN_SPREAD_MAX),
                      settle_s=SETTLE_S, fixation_s=FIXATION_S, trajectory_s=TRAJECTORY_S)
        if session in CAPTURE_PROTOCOLS:
            config["capture_protocol"] = CAPTURE_PROTOCOLS[session]
        folder.mkdir(parents=True, exist_ok=True)
    with Session9(folder, config) as store:
        if store.config["contract"] != VERSION or store.config["feature_contract"] != FEATURE_VERSION:
            raise RuntimeError("Contrato incompatible con la sesión.")
        if (args.session in CAPTURE_PROTOCOLS and
                (store.config.get("capture_protocol") != CAPTURE_PROTOCOLS[session] or
                 store.config["plan"] != plan(session))):
            raise RuntimeError(f"La sesión existente usa otro protocolo de {session}. Puede exportarse por separado.")
        if args.distance_cm is not None and args.distance_cm != store.config["distance_cm"]:
            parser.error("La distancia declarada es fija por sesión.")
        try:
            if args.session:
                capture_session(store)
        finally:
            print(f"Exportación: {store.export()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, cv2.error) as exc:
        sys.exit(str(exc))
