"""Plan de captura y controles de calidad del Tracker."""

import random

import numpy as np

VERSION = "headcontrolnet-tracker-capture-v2"
TR01_PROTOCOL = "tr01-natural-v1"
TR02_PROTOCOL = "tr02-center-checks-v1"
TR03_PROTOCOL = "tr03-natural-v1"
CONDITIONS = ("translation", "rotation", "combined")
SPLITS = {"TR-01": "train", "TR-02": "validation", "TR-03": "test"}
ANCHORS = [(f"P{i}", x, y) for i, (x, y) in enumerate(
    ((x, y) for y in (.08, .5, .92) for x in (.08, .5, .92)), 1)]
VALIDATION = [(f"I{i}", x, y) for i, (x, y) in enumerate(
    ((.29, .29), (.71, .29), (.29, .71), (.71, .71)), 1)]
TEST = [(f"F{i}", x, y) for i, (x, y) in enumerate((
    (.22, .22), (.78, .22), (.22, .78), (.78, .78),
    (.36, .50), (.64, .50), (.50, .36), (.50, .64)), 1)]
TRAJECTORIES = (
    ("H", (.15, .50), (.85, .50)),
    ("V", (.50, .15), (.50, .85)),
    ("D1", (.18, .18), (.82, .82)),
    ("D2", (.82, .18), (.18, .82)),
)
SETTLE_S = .8
FIXATION_S = 2.0
TRAJECTORY_S = 5.0
QUALITY = dict(min_frames=12, min_valid_fraction=.70, compliance_fraction=.80,
               neutral_angle_deg=5., neutral_center=.12, minimum_angle_deg=3., minimum_center=.05,
               stable_angle_spread_deg=5., stable_center_spread=.10, minimum_trajectory_bins=4)
STABILITY_FIELDS = ("yaw_rel_deg", "pitch_rel_deg", "roll_rel_deg", "face_center_dx", "face_center_dy")


def label(x, y, width, height):
    px, py = round(x * (width - 1)), round(y * (height - 1))
    return dict(target_px_x=px, target_px_y=py, target_x=px / (width - 1), target_y=py / (height - 1))


def target_for(trial, progress, width, height):
    start, end = trial["start"], trial["end"]
    xy = [a + progress * (b - a) for a, b in zip(start, end)]
    return label(*xy, width, height)


def plan(session):
    if session == "TR-01":
        rng = random.Random(9410)
        outer = [point for point in ANCHORS if point[0] != "P5"]
        visits = []
        for _ in range(3):
            order = rng.sample(outer, len(outer))
            if visits and visits[-1] == order[0]:
                order[0], order[1] = order[1], order[0]
            visits.extend(order)
        for position in (1, 7, 13, 19, 25, 30):
            visits.insert(position - 1, ANCHORS[4])
        return [
            dict(repetition_id=f"R{i:02d}", target_id=key, condition="natural", task="fixation",
                 center_control=key == "P5", start=[x, y], end=[x, y])
            for i, (key, x, y) in enumerate(visits, 1)
        ]
    if session == "TR-03":
        rng = random.Random(9412)
        visits = []
        for _ in range(3):
            fixed = [(key, (x, y), (x, y), "fixation", False) for key, x, y in TEST]
            paths = [(key, a, b, "trajectory", False) for key, a, b in TRAJECTORIES]
            rng.shuffle(fixed)
            rng.shuffle(paths)
            visits.extend(fixed + paths)
        for position in (1, 10, 16, 25, 33, 42):
            visits.insert(position - 1, ("P5", (.5, .5), (.5, .5), "fixation", True))
        return [
            dict(repetition_id=f"R{i:02d}", target_id=key, condition="natural",
                 task=task, center_control=center, start=list(a), end=list(b))
            for i, (key, a, b, task, center) in enumerate(visits, 1)
        ]
    points = {"TR-02": [*ANCHORS, *VALIDATION]}[session]
    shift = list(SPLITS).index(session)
    order = CONDITIONS[shift:] + CONDITIONS[:shift]
    rng = random.Random(9410 + shift)
    trials = []
    for condition in order:
        batch = [(key, (x, y), (x, y), "fixation", key in ("P5", "P5B"))
                 for key, x, y in [*points, ("P5B", .5, .5)]]
        if session == "TR-02":
            centers = [trial for trial in batch if trial[-1]]
            batch = [trial for trial in batch if not trial[-1]]
            rng.shuffle(batch)
            # Dos controles por bloque. Se conserva el objetivo y la condición de cada fijación.
            for position, center in zip(((1, 10), (2, 11), (5, 14))[order.index(condition)], centers):
                batch.insert(position - 1, center)
        else:
            rng.shuffle(batch)
        base = len(trials)
        trials.extend(dict(repetition_id=f"R{base + i:02d}", target_id=key, condition=condition,
                           task=task, center_control=center, start=list(a), end=list(b))
                      for i, (key, a, b, task, center) in enumerate(batch, 1))
    if session == "TR-02":
        for trial in trials:
            if trial["center_control"]:
                trial["center_drift_check"] = True
    return trials


def natural_quality(rows, total, moving=False):
    fraction = len(rows) / max(total, 1)
    sufficient = len(rows) >= QUALITY["min_frames"] and fraction >= QUALITY["min_valid_fraction"]
    summary = dict(reason="ok" if sufficient else "insufficient_detection",
                   valid_fraction=fraction, stable=None)
    if not rows:
        return summary
    values = np.array([[r[k] for k in STABILITY_FIELDS] for r in rows])
    if not np.isfinite(values).all():
        return dict(reason="nonfinite_features", valid_fraction=fraction)
    low, median, high = np.percentile(values, (10, 50, 90), axis=0)
    spread = high - low
    stable = (max(spread[:3]) <= QUALITY["stable_angle_spread_deg"] and
              max(spread[3:]) <= QUALITY["stable_center_spread"])
    summary.update(stable=bool(stable) if sufficient and not moving else None,
                   median=dict(zip(STABILITY_FIELDS, median.tolist())),
                   p90_p10=dict(zip(STABILITY_FIELDS, spread.tolist())))
    if moving:
        bins = {min(int(r["target_progress"] * 5), 4) for r in rows}
        summary["trajectory_bins"] = len(bins)
        if summary["reason"] == "ok" and len(bins) < QUALITY["minimum_trajectory_bins"]:
            summary["reason"] = "trajectory_incomplete"
        return summary
    if summary["reason"] == "ok" and not stable:
        summary["reason"] = "unstable_fixation"
    return summary


def assess(trial, rows, total):
    if trial["condition"] == "natural" or trial.get("center_drift_check"):
        return natural_quality(rows, total, trial["task"] == "trajectory")
    cfg = QUALITY
    fraction = len(rows) / max(total, 1)
    if len(rows) < cfg["min_frames"] or fraction < cfg["min_valid_fraction"]:
        return dict(reason="insufficient_detection", valid_fraction=fraction)
    values = np.array([[r[k] for k in ("yaw_rel_deg", "pitch_rel_deg", "roll_rel_deg",
                                       "face_center_dx", "face_center_dy")] for r in rows])
    if not np.isfinite(values).all():
        return dict(reason="nonfinite_features", valid_fraction=fraction)
    angles = np.max(np.abs(values[:, :3]), axis=1)
    centers = np.linalg.norm(values[:, 3:], axis=1)
    angle90, center90 = map(float, (np.percentile(angles, 90), np.percentile(centers, 90)))
    angle_neutral = float(np.mean(angles <= cfg["neutral_angle_deg"]))
    center_neutral = float(np.mean(centers <= cfg["neutral_center"]))
    spread = np.ptp(np.percentile(values, (10, 90), axis=0), axis=0)
    has_angle = angle90 >= cfg["minimum_angle_deg"]
    has_center = center90 >= cfg["minimum_center"]
    summary = dict(reason="ok", valid_fraction=fraction, angle_p90_deg=angle90, center_p90=center90,
                   angle_neutral_fraction=angle_neutral, center_neutral_fraction=center_neutral,
                   angular_spread_deg=float(max(spread[:3])), center_spread=float(max(spread[3:])),
                   observed="mixed" if has_angle and has_center else "rotation" if has_angle else
                   "translation" if has_center else "near_neutral")
    required = cfg["compliance_fraction"]
    cond = trial["condition"]
    if trial["task"] == "trajectory":
        bins = {min(int(r["target_progress"] * 5), 4) for r in rows}
        summary["trajectory_bins"] = len(bins)
        if len(bins) < cfg["minimum_trajectory_bins"]:
            summary["reason"] = "trajectory_incomplete"
            return summary
    elif max(spread[:3]) > cfg["stable_angle_spread_deg"] or max(spread[3:]) > cfg["stable_center_spread"]:
        summary["reason"] = "unstable_fixation"
        return summary
    if trial["center_control"]:
        if min(angle_neutral, center_neutral) < required:
            summary["reason"] = "center_not_neutral"
    elif cond == "translation" and (angle_neutral < required or not has_center):
        summary["reason"] = "translation_mismatch"
    elif cond == "rotation" and (center_neutral < required or not has_angle):
        summary["reason"] = "rotation_mismatch"
    elif cond == "combined" and not (has_angle and has_center):
        summary["reason"] = "combined_mismatch"
    return summary
