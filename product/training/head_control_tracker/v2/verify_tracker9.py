"""Comprueba geometría, protocolo y un CSV técnico. No entrena modelos."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from features9 import CENTER, FEATURES, RAW, face_center, neutral_reference, relative, vector
from features import geometry, pose_angles
from protocol import CONDITIONS, SPLITS, assess, plan, target_for


def synthetic_row(dx=0., dy=0., yaw=0.):
    face = {33: (-80, 0, 0), 263: (80, 0, 0), 1: (4, 45, 35), 152: (1, 115, 10)}
    rotation, _ = cv2.Rodrigues(np.deg2rad([0., yaw, 0.]))
    points = [None] * 264
    for i, xyz in face.items():
        x, y = (rotation @ xyz)[:2] + (640 + dx, 330 + dy)
        points[i] = SimpleNamespace(x=x / 1280, y=y / 720)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (dx / 100, dy / 100, -50)
    sample = SimpleNamespace(face_landmarks=[points], facial_transformation_matrixes=[matrix])
    angles, status = pose_angles(sample)
    assert status == "ok_approximate"
    offsets, size = geometry(points)
    return dict(zip((*RAW, *CENTER), (*angles, *offsets, *face_center(points, 1280, 720))),
                interocular_px=size, valid_detection=1)


def verify_simulation():
    ref = neutral_reference([synthetic_row()] * 12)
    examples = {
        "translation": (synthetic_row(dx=24, dy=-16), "translation"),
        "rotation": (synthetic_row(yaw=12), "rotation"),
        "combined": (synthetic_row(dx=24, dy=-16, yaw=12), "mixed"),
    }
    vectors = {k: vector(row, ref) for k, (row, _) in examples.items()}
    assert tuple(vectors["translation"]) == FEATURES
    assert np.allclose(list(vectors["translation"].values()), (0, 0, 0, 0, 0, 0, 0, .15, -.10))
    assert np.allclose([vectors["rotation"][k] for k in FEATURES[-2:]], (0, 0))
    assert np.allclose([vectors["combined"][k] for k in FEATURES[-2:]], (.15, -.10))
    assert examples["rotation"][0]["interocular_px"] < ref["interocular_px"]
    assert np.allclose([vectors["rotation"][k] for k in FEATURES[:7]],
                       [vectors["combined"][k] for k in FEATURES[:7]])
    for condition, (_, expected) in examples.items():
        trial = dict(condition=condition, center_control=False, task="fixation")
        result = assess(trial, [vectors[condition]] * 16, 16)
        assert result["reason"] == "ok" and result["observed"] == expected
        print(f"{condition}: {expected} aceptado")
    for condition, wrong in (("translation", "rotation"), ("rotation", "translation")):
        trial = dict(condition=condition, center_control=False, task="fixation")
        assert assess(trial, [vectors[wrong]] * 16, 16)["reason"] == f"{condition}_mismatch"
    trial = dict(condition="combined", center_control=False, task="fixation")
    assert assess(trial, [vector(synthetic_row(), ref)] * 16, 16)["reason"] == "combined_mismatch"
    assert assess(trial, [vectors["translation"]] * 16, 16)["reason"] == "combined_mismatch"
    assert assess(trial, [vectors["combined"]] * 16, 40)["reason"] == "insufficient_detection"
    assert assess(trial, [vectors["combined"]] * 8 + [vector(synthetic_row(), ref)] * 8, 16)[
        "reason"] == "unstable_fixation"
    assert assess(dict(trial, center_control=True), [vectors["combined"]] * 16, 16)[
        "reason"] == "center_not_neutral"
    path = dict(trial, task="trajectory")
    assert assess(path, [dict(vectors["combined"], target_progress=.1)] * 16, 16)[
        "reason"] == "trajectory_incomplete"
    assert assess(path, [dict(vectors["combined"], target_progress=i / 16) for i in range(16)], 16)[
        "reason"] == "ok"
    try:
        neutral_reference([synthetic_row()] * 8 + [synthetic_row(dx=30)] * 8)
    except ValueError:
        pass
    else:
        raise AssertionError("Una referencia neutral inestable fue aceptada.")


def verify_plan():
    plans = {session: plan(session) for session in SPLITS}
    previous = set()
    for session, count in zip(SPLITS, (30, 42, 42)):
        trials = plans[session]
        assert len(trials) == len({row["repetition_id"] for row in trials}) == count
        if session == "TR-01":
            assert all(t["condition"] == "natural" and t["task"] == "fixation" for t in trials)
            assert [i for i, t in enumerate(trials, 1) if t["center_control"]] == [1, 7, 13, 19, 25, 30]
            assert Counter(t["target_id"] for t in trials) == {
                f"P{i}": 6 if i == 5 else 3 for i in range(1, 10)
            }
            ordinary = [t for t in trials if not t["center_control"]]
            for trial in (trials[0], ordinary[0]):
                mixed = dict(yaw_rel_deg=22., pitch_rel_deg=-14., roll_rel_deg=9.,
                             face_center_dx=.4, face_center_dy=-.5)
                quality = assess(trial, [mixed] * 16, 16)
                assert quality["reason"] == "ok" and quality["stable"]
                assert quality["median"] == mixed
                unstable = [mixed] * 8 + [dict(mixed, face_center_dx=.7)] * 8
                assert assess(trial, unstable, 16)["reason"] == "unstable_fixation"
                assert assess(trial, [mixed] * 12, 25)["reason"] == "insufficient_detection"
        elif session == "TR-02":
            for condition in CONDITIONS:
                subset = [row for row in trials if row["condition"] == condition]
                assert len(subset) == count // 3
                assert sum(row["center_control"] for row in subset) == 2
        if session == "TR-02":
            assert [i for i, t in enumerate(trials, 1) if t["center_control"]] == [1, 10, 16, 25, 33, 42]
            for condition in CONDITIONS:
                subset = [t for t in trials if t["condition"] == condition]
                assert {t["target_id"] for t in subset} == {
                    *(f"P{i}" for i in range(1, 10)), "P5B", *(f"I{i}" for i in range(1, 5)),
                }
            for trial in trials:
                assert bool(trial.get("center_drift_check")) == trial["center_control"]
            drifted = dict(yaw_rel_deg=8., pitch_rel_deg=-7., roll_rel_deg=1.,
                           face_center_dx=-.13, face_center_dy=.22)
            assert assess(trials[0], [drifted] * 16, 16)["reason"] == "ok"
            unstable = [drifted] * 8 + [dict(drifted, face_center_dy=.45)] * 8
            assert assess(trials[0], unstable, 16)["reason"] == "unstable_fixation"
        static = {tuple(t["start"]) for t in trials if t["task"] == "fixation" and not t["center_control"]}
        if session == "TR-03":
            assert len(static) == 8 and not previous.intersection(static)
            assert all(t["condition"] == "natural" for t in trials)
            assert [i for i, t in enumerate(trials, 1) if t["center_control"]] == [1, 10, 16, 25, 33, 42]
            assert Counter(t["target_id"] for t in trials if t["task"] == "fixation") == {
                "P5": 6, **{f"F{i}": 3 for i in range(1, 9)},
            }
            paths = [t for t in trials if t["task"] == "trajectory"]
            assert Counter(t["target_id"] for t in paths) == {"H": 3, "V": 3, "D1": 3, "D2": 3}
            for trial in paths:
                assert len({tuple(target_for(trial, i / 20, 1920, 1080).values()) for i in range(21)}) == 21
            moving = [dict(yaw_rel_deg=i * 3., pitch_rel_deg=-i * 2., roll_rel_deg=0.,
                           face_center_dx=i * .08, face_center_dy=0., target_progress=i / 20)
                      for i in range(20)]
            assert assess(paths[0], moving, 20)["reason"] == "ok"
            assert assess(paths[0], [dict(r, target_progress=.1) for r in moving], 20)[
                "reason"] == "trajectory_incomplete"
        previous.update(static)
    print("Plan validado: 30/42/42, anclas reservadas y 12 trayectorias.")


def verify_camera(folder):
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["split"] == "technical" and meta["error"] is None
    ref = meta["neutral"]
    width, height = meta["screen_size"]
    with (folder / "fotogramas_tecnicos.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid = [r for r in rows if r["valid_detection"] == "1"]
    assert valid and len({r["timestamp_ms"] for r in rows}) == len(rows)
    assert tuple(meta["camera"]["features"]) == FEATURES
    for row in valid:
        assert (int(row["camera_width"]), int(row["camera_height"])) == (1280, 720)
        assert (int(row["screen_width"]), int(row["screen_height"])) == (width, height)
        assert float(row["target_presented_monotonic_s"]) <= float(row["frame_monotonic_s"])
        assert float(row["target_x"]) == int(row["target_px_x"]) / (width - 1)
        assert float(row["target_y"]) == int(row["target_px_y"]) / (height - 1)
        actual = {key: float(row[key]) for key in (*RAW, *CENTER)}
        values = vector(actual, ref)
        assert all(np.isfinite(value) and float(row[key]) == value for key, value in values.items())
        assert all(values[key] == value for key, value in relative([actual[k] for k in RAW], ref["raw"]).items())
    print(f"Cámara: {len(valid)}/{len(rows)} detecciones y nueve entradas verificadas.")
    print(f"Pantalla: {width}x{height}, neutral: {ref['count']} detecciones, "
          f"distancia interocular: {ref['interocular_px']:.2f} px.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical", type=Path)
    options = parser.parse_args()
    verify_simulation()
    verify_plan()
    if options.technical:
        verify_camera(options.technical)


if __name__ == "__main__":
    main()
