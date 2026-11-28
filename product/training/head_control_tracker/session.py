"""Plan, etiquetas y exportación del dataset de regresión."""

from collections import Counter
from hashlib import sha256
import json
import random
from uuid import uuid4

from storage import ROW_FIELDS, Store, dumps, write_csv

CONTRACT = "headcontrolnet-tracker-capture-v1"
TRIAL_FIELDS = ("repetition_id", "round", "target_id", "target_x", "target_y",
                "target_px_x", "target_px_y", "screen_width", "screen_height")


def make_plan(session_id, columns, rounds, width, height):
    points = []
    for row, fy in enumerate((.08, .50, .92)):
        for col in range(columns):
            fx = .08 + .84 * col / (columns - 1)
            x, y = round(fx * (width - 1)), round(fy * (height - 1))
            points.append(dict(target_id=f"P{row * columns + col + 1}", target_x=x / (width - 1),
                               target_y=y / (height - 1), target_px_x=x, target_px_y=y,
                               screen_width=width, screen_height=height))
    rng = random.Random(int.from_bytes(sha256(session_id.encode()).digest(), "big"))
    plan = []
    for number in range(1, rounds + 1):
        batch = rng.sample(points, len(points))
        if plan and batch[0]["target_id"] == plan[-1]["target_id"]:
            batch[0], batch[1] = batch[1], batch[0]
        plan.extend(dict(p, repetition_id=f'R{number:02d}-{p["target_id"]}', round=number)
                    for p in batch)
    return plan


class TrackerStore(Store):
    def attempt(self, trial, baseline, status, rows, quality):
        if trial not in self.config["plan"]:
            raise ValueError("La fijación no coincide con el plan de la sesión.")
        super().attempt(trial, baseline, status, rows, quality)

    def export(self):
        with self.db:
            self.db.execute("BEGIN")
            attempts = list(self.db.execute("SELECT id, status, baseline, data FROM attempts ORDER BY id"))
            baselines = {key: json.loads(data) for key, data in self.db.execute("SELECT id, data FROM baselines")}
        folder = self.folder / f"export_{uuid4().hex[:12]}"
        folder.mkdir()
        rows, audit = [], []
        counts = dict.fromkeys((trial["target_id"] for trial in self.config["plan"]), 0)
        accepted = Counter()
        session = {key: self.config[key] for key in ("session_id", "split", "distance_cm")}
        for attempt_id, status, baseline, data in attempts:
            record = json.loads(data)
            fields = dict(**session, **record["trial"], attempt_id=attempt_id, baseline_id=baseline)
            audit.append(dict(fields, status=status, timestamp_utc=record["timestamp_utc"],
                              valid_samples=record["valid_samples"], quality=dumps(record["quality"])))
            if status == "accepted":
                target = fields["target_id"]
                accepted[target] += 1
                counts[target] += record["valid_samples"]
                rows.extend(dict(fields, sample=i, **row) for i, row in enumerate(record["rows"], 1))
        fields = (*session, *TRIAL_FIELDS, "attempt_id", "baseline_id")
        name = "fotogramas_tecnicos.csv" if self.config["split"] == "technical" else "dataset.csv"
        write_csv(folder / name, (*fields, "sample", *ROW_FIELDS), rows)
        write_csv(folder / "intentos.csv", (*fields, "status", "timestamp_utc", "valid_samples", "quality"), audit)
        metadata = dict(self.config, baselines=baselines, samples_by_target=counts,
                        repetitions_by_target=dict.fromkeys(counts, 0) | accepted,
                        complete=sum(accepted.values()) == len(self.config["plan"]))
        (folder / "metadata.json").write_text(dumps(metadata) + "\n", encoding="utf-8")
        return folder
