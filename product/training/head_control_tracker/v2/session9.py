"""Persistencia del protocolo, aislada de cualquier captura anterior."""

from collections import Counter
import json
from math import isfinite
from uuid import uuid4

from features9 import CENTER, FEATURES, RAW
from protocol import assess, target_for
from storage import ROW_FIELDS, Store, dumps, utc, write_csv

LABELS = ("target_px_x", "target_px_y", "target_x", "target_y")
GROUP_FIELDS = ("session_id", "split", "distance_cm", "screen_width", "screen_height",
                "repetition_id", "target_id", "condition", "task", "center_control", "attempt_id", "baseline_id")
FRAME_FIELDS = ("target_progress", "target_presented_monotonic_s", "frame_monotonic_s",
                *LABELS, *ROW_FIELDS, *CENTER, *FEATURES[-2:])


class Session9(Store):
    extra_group_fields = ()
    extra_frame_fields = ()

    def reference(self):
        references = list(self.db.execute("SELECT id, data FROM baselines"))
        if not references:
            return None
        if len(references) != 1:
            raise RuntimeError("La sesión debe tener una sola referencia neutral.")
        key, payload = references[0]
        data = json.loads(payload)
        return key, dict(data["stats"], raw=[data["values"][k] for k in RAW]), data["camera"]

    def attempt(self, trial, baseline, status, rows, quality):
        if trial not in self.config["plan"]:
            raise ValueError("Intento ajeno al plan de la sesión.")
        if status == "accepted":
            if (quality["reason"] != "ok" or assess(trial, rows, sum(quality["counts"].values()))["reason"] != "ok"
                    or len({r["timestamp_ms"] for r in rows}) != len(rows)):
                raise ValueError("Intento aceptado sin cumplir las condiciones de calidad.")
            width, height = self.config["screen_size"]
            for row in rows:
                if (row["valid_detection"] != 1 or not all(isfinite(row[k]) for k in FEATURES)
                        or not 0 <= row["target_progress"] <= 1
                        or not row["target_presented_monotonic_s"] <= row["frame_monotonic_s"]
                        or any(row[k] != v for k, v in target_for(trial, row["target_progress"], width, height).items())):
                    raise ValueError("El fotograma no coincide con la etiqueta o el contrato del intento.")
        data = dict(trial=trial, rows=rows if status == "accepted" else [], valid_samples=len(rows),
                    quality=quality, timestamp_utc=utc())
        with self.db:
            self.db.execute("INSERT INTO attempts(repetition, status, baseline, data) VALUES (?, ?, ?, ?)",
                            (trial["repetition_id"], status, baseline, dumps(data)))

    def export(self):
        with self.db:
            self.db.execute("BEGIN")
            attempts = list(self.db.execute("SELECT id, status, baseline, data FROM attempts ORDER BY id"))
            baselines = {key: json.loads(value) for key, value in self.db.execute("SELECT id, data FROM baselines")}
        folder = self.folder / f"export_{uuid4().hex[:12]}"
        folder.mkdir()
        session = {k: self.config[k] for k in ("session_id", "split", "distance_cm")}
        width, height = self.config["screen_size"]
        rows, audit, coverage = [], [], Counter()
        for attempt_id, status, baseline, payload in attempts:
            attempt = json.loads(payload)
            trial = attempt["trial"]
            group = dict(**session, screen_width=width, screen_height=height, attempt_id=attempt_id,
                         baseline_id=baseline, **{k: trial[k] for k in ("repetition_id", "target_id", "condition",
                                                                       "task", "center_control", *self.extra_group_fields)})
            audit.append(dict(group, status=status, timestamp_utc=attempt["timestamp_utc"],
                              valid_samples=attempt["valid_samples"], quality=dumps(attempt["quality"])))
            if status == "accepted":
                coverage[f'{trial["condition"]}/{trial["task"]}'] += 1
                rows.extend(dict(group, sample=i, **row) for i, row in enumerate(attempt["rows"], 1))
        group_fields = (*GROUP_FIELDS, *self.extra_group_fields)
        write_csv(folder / "dataset.csv", (*group_fields, "sample", *FRAME_FIELDS, *self.extra_frame_fields), rows)
        write_csv(folder / "intentos.csv", (*group_fields, "status", "timestamp_utc", "valid_samples", "quality"), audit)
        metadata = dict(self.config, baselines=baselines, accepted_by_condition_and_task=dict(coverage),
                        complete=sum(coverage.values()) == len(self.config["plan"]))
        (folder / "metadata.json").write_text(dumps(metadata) + "\n", encoding="utf-8")
        return folder
