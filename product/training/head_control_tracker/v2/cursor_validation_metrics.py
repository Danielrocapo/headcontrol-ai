"""Métricas del puntero leído de Windows, con igual peso por prueba."""

from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

JUMP_PX = 75.


def measure(rows, kind):
    valid = [r for r in rows if r["detection"]]
    active = [r for r in rows if r["actuated"]]
    if not active:
        return None
    target = np.array([[r["target_x"], r["target_y"]] for r in active])
    cursor = np.array([[r["cursor_x"], r["cursor_y"]] for r in active])
    delta = cursor - target
    e = np.linalg.norm(delta, axis=1)
    residual_steps, cursor_steps, reacquisition = [], [], []
    valid_gap = loss_seen = False
    detected_since = None
    recovery_delays = []
    for i, r in enumerate(rows):
        if not r["detection"]:
            valid_gap = loss_seen = True
            detected_since = None
        elif valid_gap and detected_since is None:
            detected_since = r["elapsed_s"]
        if r["actuated"] and valid_gap:
            reacquisition.append(r["cursor_step_px"])
            recovery_delays.append((r["elapsed_s"] - detected_since) * 1000)
            valid_gap = False
        if i and r["actuated"] and rows[i-1]["actuated"]:
            before = rows[i-1]
            residual_steps.append(np.hypot(r["cursor_x"]-before["cursor_x"]-r["target_x"]+before["target_x"],
                                           r["cursor_y"]-before["cursor_y"]-r["target_y"]+before["target_y"]))
        if r["actuated"]:
            cursor_steps.append(r["cursor_step_px"])
    times = [r["elapsed_s"] for r in rows]
    latency = [r["inference_ms"] for r in valid]
    result = dict(frames=len(rows), active_frames=len(active), lost_frames=len(rows)-len(valid),
                  outside_frames=sum(r["outside"] for r in valid),
                  fps=(len(times)-1)/(times[-1]-times[0]) if times[-1] > times[0] else None,
                  inference_median_ms=float(np.median(latency)),
                  inference_p90_ms=float(np.percentile(latency,90)), mean_px=float(e.mean()),
                  median_px=float(np.median(e)), p90_px=float(np.percentile(e,90)),
                  mae_x_px=float(np.abs(delta[:,0]).mean()), mae_y_px=float(np.abs(delta[:,1]).mean()),
                  jitter_residual_p90_px=float(np.percentile(residual_steps,90)) if residual_steps else None,
                  jumps=sum(step >= JUMP_PX for step in cursor_steps), max_step_px=max(cursor_steps),
                  recovery_step_px=max(reacquisition) if reacquisition else None,
                  recovery_after_detection_min_ms=min(recovery_delays) if recovery_delays else None,
                  longest_loss_s=longest_loss(rows))
    if kind == "reposo":
        n = max(1, len(cursor)//5)
        drift = np.median(cursor[-n:],axis=0)-np.median(cursor[:n],axis=0)
        jitter = np.linalg.norm(cursor-np.median(cursor,axis=0),axis=1)
        result.update(rest_drift_x_px=float(drift[0]),rest_drift_y_px=float(drift[1]),
                      rest_jitter_p90_px=float(np.percentile(jitter,90)))
    if kind == "recuperacion":
        result["recovery_observed"] = bool(loss_seen and reacquisition)
    return result


def longest_loss(rows):
    start = None
    longest = 0.
    for r in rows:
        if not r["detection"] and start is None:
            start = r["elapsed_s"]
        if start is not None:
            longest = max(longest, r["elapsed_s"]-start)
            if r["detection"]:
                start = None
    return longest


def summarize(results):
    by_task = defaultdict(set)
    for r in results:
        by_task[r["task"]].add(r["mode"])
    comparable = {task for task, modes in by_task.items() if modes == {"RAW","CALIBRATED"}
                  and task != "recuperacion"}
    summaries = []
    for mode in ("RAW","CALIBRATED"):
        group = [r for r in results if r["mode"] == mode and r["task"] in comparable]
        if not group:
            continue
        fields = ("mean_px","median_px","p90_px","mae_x_px","mae_y_px","jitter_residual_p90_px",
                  "fps","inference_median_ms","inference_p90_ms")
        summaries.append(dict(mode=mode, paired_trials=len(group),
            **{k:float(np.mean([r[k] for r in group if r[k] is not None]))
               if any(r[k] is not None for r in group) else None for k in fields},
            **{k:sum(r[k] for r in group) for k in ("frames","lost_frames","outside_frames","jumps")}))
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle,list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def report(folder, rows, attempts):
    confirmed = {a["attempt"]: a for a in attempts if a["outcome"] == "confirmed"}
    grouped = defaultdict(list)
    for row in rows:
        if row["attempt"] in confirmed and row["phase"] == "measure":
            grouped[row["attempt"]].append(row)
    results = []
    for attempt, frames in grouped.items():
        info = confirmed[attempt]
        computed = measure(frames,info["task"])
        if computed:
            results.append(dict(info,**computed))
    summaries = summarize(results)
    write_csv(folder/"fotogramas.csv",rows)
    write_csv(folder/"intentos.csv",attempts)
    write_csv(folder/"metricas_por_prueba.csv",results)
    write_csv(folder/"comparacion_pareada.csv",summaries)
    lines=["# Tracker D · Validación física del cursor", "",
           f"Se obtuvieron métricas de {len(results)}/18 pruebas confirmadas.", "",
           "Las métricas usan la posición leída del puntero de Windows. El resumen promedia cada métrica por prueba "
           "con igual peso y solo usa pares completos. La recuperación se informa por separado.", "",
           "| Modo | Pares | Medio | Mediana | P90 | MAE X/Y | FPS | Inferencia mediana | Fuera | Pérdidas | Saltos |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['mode']} | {r['paired_trials']} | {r['mean_px']:.1f} | {r['median_px']:.1f} | "
                     f"{r['p90_px']:.1f} | {r['mae_x_px']:.1f} / {r['mae_y_px']:.1f} | {r['fps']:.1f} | "
                     f"{r['inference_median_ms']:.2f} ms | {r['outside_frames']} | {r['lost_frames']} | {r['jumps']} |")
    if not summaries:
        lines += ["", "La captura todavía no contiene pares confirmados para comparar los dos modos."]
    lines += ["", "## Reposo y recuperación", ""]
    for r in results:
        if r["task"] == "reposo":
            lines.append(f"- {r['mode']} en reposo. Jitter P90 {r['rest_jitter_p90_px']:.1f} px y deriva "
                         f"X {r['rest_drift_x_px']:+.1f} px / Y {r['rest_drift_y_px']:+.1f} px.")
        if r["task"] == "recuperacion":
            lines.append(f"- {r['mode']} al recuperar. Pérdida máxima {r['longest_loss_s']:.2f} s y "
                         f"primer paso máximo {r['recovery_step_px']:.1f} px.")
    if not any(r["task"] in ("reposo", "recuperacion") for r in results):
        lines.append("Quedan pendientes las pruebas confirmadas de reposo y recuperación.")
    lines+=["", "Los errores se expresan en píxeles. Un salto es un paso de 75 px o más entre la lectura previa "
                 "y la posterior a un envío. El jitter dinámico es el P90 del cambio del residual entre fotogramas "
                 "actuados consecutivos. En reposo se mide dispersión respecto a la mediana y deriva entre los "
                 "quintiles inicial y final. La latencia incluye inferencia y normalización, sin latencia extremo a extremo."]
    links = (("Métricas por prueba", "metricas_por_prueba.csv"), ("Resumen pareado", "comparacion_pareada.csv"),
             ("Fotogramas", "fotogramas.csv"), ("Intentos", "intentos.csv"))
    lines += ["", " · ".join(f"[{label}]({name})" for label,name in links if (folder/name).is_file())]
    (folder/"INFORME_CURSOR.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return results
