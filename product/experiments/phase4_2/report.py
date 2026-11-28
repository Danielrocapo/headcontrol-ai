"""Compara las capturas de postura de fase 4.2 con el desarrollo de la MLP 9."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

TRAINING_CAMERA = (1280, 720)
NEAR_LIMIT_DEG = 2.0
CONDITIONS = {
    "neutral": ("Neutral", (-2, 2)),
    "up_light": ("Arriba leve", (-6, -3)),
    "up_moderate": ("Arriba moderado", (-12, -8)),
}
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))
from phase4_models import POSE_FORMULA, RUN3

sys.path.insert(0, str(ROOT / "product" / "training"))
from audit_datasets import inspect_dataset


def training_pitch():
    path = RUN3 / "head_pose_mlp_9.json"
    contents = path.read_bytes()
    meta = json.loads(contents)
    extraction = meta["extraction"]
    if (meta["model_id"] != "head_pose_mlp_9" or meta["features"].index("head_pitch_deg") != 7
            or extraction["pose_units"] != "degrees" or extraction["pose_formula"] != POSE_FORMULA):
        raise ValueError("Metadatos de postura incompatibles con la fase 4.2.")
    eligible, source = inspect_dataset("head_pose")
    splits = json.loads((RUN3 / meta["split_manifest"]).read_text(encoding="utf-8"))
    held = set(next(split["held_fixations"] for split in splits if split["dataset"] == "head_pose"))
    development = [row for row in eligible if row["fixation"] not in held]
    if (source["sha256"] != meta["dataset_sha256"] or source["path"] != meta["dataset"]
            or {r["fixation"] for r in eligible if r["split"] == "reserved"} != held
            or len(development) != meta["development_frames"]
            or len({r["fixation"] for r in development}) != meta["development_fixations"]):
        raise ValueError("No se pudo reproducir la particion de desarrollo de la MLP 9.")
    pitches = np.array([float(row["head_pitch_deg"]) for row in development])
    reference = dict(min=float(pitches.min()), max=float(pitches.max()), median=float(np.median(pitches)),
                     frames=len(pitches), fixations=meta["development_fixations"],
                     model_version=meta["model_version"], metadata_sha256=sha256(contents).hexdigest(),
                     dataset_sha256=source["sha256"], split_sha256=sha256(
                         (RUN3 / meta["split_manifest"]).read_bytes()).hexdigest())
    return reference, pitches


def coverage(pitch, training):
    lo, hi = training["min"], training["max"]
    if not lo <= pitch <= hi:
        return "outside"
    return "near_limit" if min(pitch - lo, hi - pitch) <= NEAR_LIMIT_DEG else "inside"


def series(rows, field):
    return np.array([float(row[field]) for row in rows if row["valid_detection"] == "1" and row[field]])


def stats(rows):
    angles = [series(rows, f"head_{axis}_deg") for axis in ("pitch", "yaw", "roll")]
    pitch = angles[0]
    relative = series(rows, "relative_pitch_deg")
    statuses = Counter(row["status"] for row in rows)
    counts = Counter(row["pitch_coverage"] for row in rows if row["valid_detection"] == "1")
    run = longest = 0
    for row in rows:
        run = 0 if row["valid_detection"] == "1" else run + 1
        longest = max(longest, run)
    return dict(frames=len(rows), valid=len(pitch), pitch=pitch, relative=relative, counts=counts,
                statuses=statuses, longest=longest, median=[float(np.median(a)) for a in angles] if len(pitch) else [],
                iqr=[float(np.subtract(*np.percentile(a, [75, 25]))) for a in angles] if len(pitch) else [])


def angle(value):
    return f"{value:+.2f}°" if np.isfinite(value) else "—"


def percentage(count, total):
    return f"{100 * count / total:.1f} %" if total else "—"


def save_plots(folder, training, development, groups, trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"neutral": "#277574", "up_light": "#5565aa", "up_moderate": "#b45529"}
    figure, (dist, repeated) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    collections = [("Entrenamiento", development, "#68727a")] + [
        (CONDITIONS.get(key, ("Prueba técnica",))[0], group["pitch"], palette.get(key, "#277574"))
        for key, group in groups.items() if group["valid"]
    ]
    for axes in (dist, repeated):
        axes.axvspan(training["min"], training["max"], color="#9eafba", alpha=0.15)
        axes.axvline(training["min"], color="#68727a", ls="--", lw=1)
        axes.axvline(training["max"], color="#68727a", ls="--", lw=1)
        axes.grid(axis="x", alpha=0.2)
    bins = np.linspace(min(values.min() for _, values, _ in collections) - 1,
                       max(values.max() for _, values, _ in collections) + 1, 34)
    for label, values, color in collections:
        dist.hist(values, bins=bins, weights=np.full(len(values), 100 / len(values)),
                  histtype="step", lw=1.9, label=label, color=color)
    dist.set(ylabel="% de fotogramas por intervalo", title="Distribución de pitch y rango de entrenamiento")
    dist.legend(loc="upper right")
    for number, (trial, group) in enumerate(trials, 1):
        if group["valid"]:
            q1, mid, q3 = np.percentile(group["pitch"], [25, 50, 75])
            color = palette.get(trial["condition"], "#277574")
            repeated.hlines(number, q1, q3, color=color, linewidth=3)
            repeated.plot(mid, number, "o", color=color)
    labels = [f"{trial['repetition']} · {CONDITIONS.get(trial['condition'], ('Prueba técnica',))[0]}"
              for trial, _ in trials]
    repeated.set(yticks=range(1, len(trials) + 1), yticklabels=labels, ylim=(0.4, len(labels) + 0.6),
                 xlabel="Pitch absoluto estimado (°)", title="Mediana e intervalo intercuartílico por repetición")
    repeated.invert_yaxis()
    figure.savefig(folder / "distribucion_pitch.png", dpi=150)
    plt.close(figure)


def make_report(folder):
    meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    with (folder / "fotogramas.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    training, development = training_pitch()
    if training != meta["training_pitch"] or meta["coverage_margin_deg"] != NEAR_LIMIT_DEG:
        raise ValueError("La referencia de entrenamiento difiere de la usada en la captura.")
    by_trial, by_condition = defaultdict(list), defaultdict(list)
    for row in rows:
        by_trial[int(row["trial_id"])].append(row)
        if row["condition"] != "baseline":
            by_condition[row["condition"]].append(row)
    complete = set(meta["completed_trials"])
    trials = [(trial, stats(by_trial[trial["trial_id"]])) for trial in meta["plan"]
              if trial["trial_id"] in complete]
    keys = ("smoke",) if meta["mode"] == "smoke" else tuple(CONDITIONS)
    groups = {key: stats([row for row in by_condition[key] if int(row["trial_id"]) in complete]) for key in keys}
    sizes = sorted({(int(row["camera_width"]), int(row["camera_height"])) for row in rows if row["camera_width"]})
    baseline = meta["baseline_pitch_deg"]
    reference = angle(baseline) if baseline is not None else "no registrada"
    scope = "Prueba técnica. Sus fotogramas no forman parte del protocolo." if meta["mode"] == "smoke" else (
        f"Protocolo {'completo' if meta['complete'] else 'incompleto'} con "
        f"{len(trials)}/15 repeticiones de cinco segundos completadas."
    )
    result = [
        "# HeadControl AI · Fase 4.2", "", scope, "",
        f"La referencia neutral inicial fue **{reference}**. El pitch relativo resta esa referencia fija.",
        f"La MLP 9 de fase 3 vio **{training['frames']} fotogramas elegibles** de "
        f"**{training['fixations']} fijaciones de desarrollo**. Su pitch abarca "
        f"**{angle(training['min'])} a {angle(training['max'])}**, con mediana **{angle(training['median'])}**.",
        f"Resolución efectiva registrada **{', '.join(f'{w} × {h}' for w, h in sizes) or 'sin fotogramas'}**. "
        f"La geometría de entrenamiento fue {TRAINING_CAMERA[0]} × {TRAINING_CAMERA[1]}.",
        "", "## Distribución y cobertura", "",
        f"«Cerca» indica un valor dentro del rango a {NEAR_LIMIT_DEG:g}° o menos de un extremo. "
        "«Dentro total» incluye «cerca». Los porcentajes de cobertura usan solo fotogramas válidos. "
        "Las medianas por condición dan el mismo peso a cada repetición con pose válida. "
        "Los porcentajes y el histograma usan los fotogramas disponibles.",
        "", "| Condición | Reps. útiles | Válidos / registrados | Pitch mediano | Δ neutral mediano | "
        "Dentro total | Cerca | Fuera | En banda relativa | Rango de medianas |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group in groups.items():
        useful = [s for trial, s in trials if trial["condition"] == key and s["valid"]]
        medians = [s["median"][0] for s in useful]
        outside = group["counts"]["outside"]
        n = group["valid"]
        mid = angle(np.median(medians)) if medians else "—"
        delta = angle(np.median(medians) - baseline) if medians and baseline is not None else "—"
        span = f"{angle(min(medians))} a {angle(max(medians))}" if medians else "—"
        band_rate = "—"
        if key in CONDITIONS:
            low, high = CONDITIONS[key][1]
            relative = group["relative"]
            band_rate = percentage(np.count_nonzero((relative >= low) & (relative <= high)), len(relative))
        result.append(f"| {CONDITIONS.get(key, ('Prueba técnica',))[0]} | {len(useful)} | {n}/{group['frames']} | "
                      f"{mid} | {delta} | {percentage(n - outside, n)} | "
                      f"{percentage(group['counts']['near_limit'], n)} | {percentage(outside, n)} | {band_rate} | {span} |")
    save_plots(folder, training, development, groups, trials)
    result += ["", "![Distribuciones y variabilidad de pitch](distribucion_pitch.png)", "",
               "## Calidad por repetición", "",
               "Los IQR son el ancho del intervalo central del 50 % de las estimaciones. Una racha cuenta "
               "fotogramas consecutivos sin pose válida. Los duplicados exactos se excluyen de la inferencia.",
               "", "| Serie | Condición | Válidos | Pitch / yaw / roll medianos | IQR pitch / yaw / roll | "
               "Racha sin pose | Motivos registrados |", "|---:|---|---:|---|---|---:|---|"]
    for trial, group in trials:
        median = " / ".join(map(angle, group["median"])) or "—"
        iqr = " / ".join(f"{x:.2f}°" for x in group["iqr"]) or "—"
        losses = ", ".join(f"`{k}` × {n}" for k, n in sorted(group["statuses"].items())
                           if k != "ok_approximate") or "—"
        result.append(f"| {trial['repetition']} | {CONDITIONS.get(trial['condition'], ('Prueba técnica',))[0]} | "
                      f"{percentage(group['valid'], group['frames'])} | {median} | {iqr} | {group['longest']} | {losses} |")
    result += ["", "## Separación, signo y próximo dataset", "",
               "La extracción usa la misma matriz facial y la misma imagen espejada de fase 4. Los ángulos "
               "se expresan en grados. Un pitch más negativo al inclinar la cabeza hacia arriba comprueba la convención "
               "en esta sesión. La cámara proporciona una estimación aproximada y el rango marginal de pitch "
               "describe cobertura observada, sin garantizar precisión de mirada."]
    by_rep = defaultdict(dict)
    for trial, group in trials:
        if group["valid"]:
            by_rep[trial["repetition"]][trial["condition"]] = group
    if meta["mode"] == "protocol":
        result += ["", "| Contraste emparejado | Pares | Δ de medianas | Orden negativo | IQR sin solapamiento y en orden |",
                   "|---|---:|---:|---:|---:|"]
        for a, b in (("neutral", "up_light"), ("up_light", "up_moderate"), ("neutral", "up_moderate")):
            pairs = [(r[a], r[b]) for r in by_rep.values() if a in r and b in r]
            changes = [right["median"][0] - left["median"][0] for left, right in pairs]
            separated = sum(np.percentile(right["pitch"], 75) < np.percentile(left["pitch"], 25)
                            for left, right in pairs)
            result.append(f"| {CONDITIONS[b][0]} menos {CONDITIONS[a][0]} | {len(pairs)} | "
                          f"{angle(np.median(changes)) if pairs else '—'} | {sum(x < 0 for x in changes)} | {separated} |")
        ordered = [all(key in rep for key in CONDITIONS)
                   and rep["up_moderate"]["median"][0] < rep["up_light"]["median"][0] < rep["neutral"]["median"][0]
                   for rep in by_rep.values()]
        if sum(ordered) == 5:
            moderate = [rep["up_moderate"] for rep in by_rep.values()]
            low = np.median([np.percentile(group["pitch"], 10) for group in moderate])
            high = np.median([np.percentile(group["pitch"], 90) for group in moderate])
            result += ["", "Las cinco series útiles ordenaron sus medianas en el sentido esperado.",
                       f"El intervalo típico P10–P90 de moderado fue **{angle(low)} a {angle(high)}** "
                       "(medianas de los percentiles por repetición). Replicar esta cobertura cómoda en sesiones "
                       "independientes. Si cruza el límite inferior, incluir fijaciones cerca del límite y "
                       "en el tramo medido que queda fuera. Revisar el solapamiento de IQR antes de fijar bandas."]
        else:
            result += ["", f"**{sum(ordered)}/5** series ordenaron las tres medianas en el sentido esperado. "
                       "Revisar las pérdidas, la dispersión y el solapamiento antes de elegir un rango nuevo."]
    else:
        result += ["", "La prueba técnica no verifica la separación de condiciones. La selección del "
                   "rango para el próximo dataset queda pendiente de las cinco repeticiones por condición."]
    output = folder / "PHASE_4_2_REPORT.md"
    output.write_text("\n".join(result) + "\n", encoding="utf-8")
    print(f"Informe: {output}")
