"""Registro transaccional de repeticiones y exportaciones de solo lectura."""

from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
import random
import sqlite3
from uuid import uuid4

from features import FEATURES, RAW, utc

CLASSES = ("CENTRO", "IZQUIERDA", "DERECHA", "ARRIBA", "ABAJO")
ROUNDS = 8
MIN_SAMPLES = 8
MAX_SAMPLES = 10
SETTLE_SECONDS = .65
MEASURE_SECONDS = 2.0
SAMPLE_INTERVAL = .18
ROW_FIELDS = ("timestamp_utc", "timestamp_ms", "camera_width", "camera_height", "status",
              "pose_status", "face_detected", "valid_detection", "interocular_px", *RAW, *FEATURES)


def make_plan(session):
    rng = random.Random(int.from_bytes(sha256(session.encode()).digest(), "big"))
    plan = []
    for round_no in range(1, ROUNDS + 1):
        order = rng.sample(CLASSES, len(CLASSES))
        if plan and order[0] == plan[-1]["label"]:
            order[0], order[1] = order[1], order[0]
        plan.extend(dict(repetition_id=f"R{round_no:02d}-{label}", round=round_no, label=label)
                    for label in order)
    return plan


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Store:
    def __init__(self, folder, config=None):
        self.folder = folder
        path = folder / "session.sqlite3"
        if config is None and not path.is_file():
            raise FileNotFoundError(path)
        self.db = sqlite3.connect(path)
        try:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT);
                CREATE TABLE IF NOT EXISTS baselines (id TEXT PRIMARY KEY, data TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, repetition TEXT NOT NULL, status TEXT NOT NULL,
                    baseline TEXT NOT NULL, data TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_accepted ON attempts(repetition) WHERE status='accepted';
            """)
            with self.db:
                if config is not None:
                    self.db.execute("INSERT INTO config VALUES (1, ?)", (dumps(config),))
            self.config = json.loads(self.db.execute("SELECT data FROM config").fetchone()[0])
        except BaseException:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def baseline(self, values, camera, stats):
        key = uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO baselines VALUES (?, ?)", (key, dumps(dict(
                values=dict(zip(RAW, map(float, values))), camera=camera, stats=stats, timestamp_utc=utc()))))
        return key

    def attempt(self, trial, baseline, status, rows, quality):
        valid = len(rows)
        if status == "accepted":
            if (not MIN_SAMPLES <= valid <= MAX_SAMPLES or
                    any(row.get("valid_detection") != 1 or
                        not all(isfinite(row[k]) for k in FEATURES) for row in rows) or
                    len({row["timestamp_ms"] for row in rows}) != valid):
                raise ValueError("La repetición no tiene suficientes muestras válidas y distintas.")
        data = dict(trial=trial, rows=rows if status == "accepted" else [], valid_samples=valid,
                    quality=dict(quality), timestamp_utc=utc())
        with self.db:
            self.db.execute("INSERT INTO attempts(repetition, status, baseline, data) VALUES (?, ?, ?, ?)",
                            (trial["repetition_id"], status, baseline, dumps(data)))

    def pending(self):
        accepted = {r[0] for r in self.db.execute("SELECT repetition FROM attempts WHERE status='accepted'")}
        return [trial for trial in self.config["plan"] if trial["repetition_id"] not in accepted]

    def export(self):
        with self.db:
            self.db.execute("BEGIN")
            attempts = list(self.db.execute("SELECT id, status, baseline, data FROM attempts ORDER BY id"))
            baselines = dict(self.db.execute("SELECT id, data FROM baselines"))
        folder = self.folder / f"export_{uuid4().hex[:12]}"
        folder.mkdir()
        rows, audit = [], []
        counts, repetitions = Counter(), Counter()
        session = {k: self.config[k] for k in ("session_id", "split", "distance_cm")}
        for attempt_id, status, baseline_id, payload in attempts:
            data = json.loads(payload)
            group = dict(**session, **data["trial"], attempt_id=attempt_id, baseline_id=baseline_id)
            audit.append(dict(group, status=status, valid_samples=data["valid_samples"],
                              timestamp_utc=data["timestamp_utc"], quality=dumps(data["quality"])))
            if status == "accepted":
                label = group["label"]
                repetitions[label] += 1
                counts[label] += data["valid_samples"]
                rows.extend(dict(group, sample=i, **row) for i, row in enumerate(data["rows"], 1))
        fields = (*session, "repetition_id", "round", "label", "attempt_id", "baseline_id")
        write_csv(folder / "dataset.csv", (*fields, "sample", *ROW_FIELDS), rows)
        write_csv(folder / "intentos.csv", (*fields, "status", "valid_samples", "timestamp_utc", "quality"), audit)
        config = dict(self.config, baselines={k: json.loads(v) for k, v in baselines.items()},
                      samples_by_class=dict.fromkeys(CLASSES, 0) | counts,
                      repetitions_by_class=dict.fromkeys(CLASSES, 0) | repetitions,
                      complete=sum(repetitions.values()) == len(self.config["plan"]))
        (folder / "metadata.json").write_text(dumps(config) + "\n", encoding="utf-8")
        return folder
