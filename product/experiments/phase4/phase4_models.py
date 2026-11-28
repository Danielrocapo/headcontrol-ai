"""Carga de solo lectura. PyTorch se ejecuta en el entorno CPU de fase 3."""

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRAINING = ROOT / "product" / "training"
RUN3 = TRAINING / "runs" / "phase3_20261124T212727_633188Z"
EYES = ("eye_a_x", "eye_a_y", "eye_a_open", "eye_b_x", "eye_b_y", "eye_b_open")
POSE = ("head_yaw_deg", "head_pitch_deg", "head_roll_deg")
TARGETS = ("target_x", "target_y")
MODELS = ("mlp_6", "mlp_9", "extra_trees_6_reference", "extra_trees_9_reference")
EYE_FORMULAS = {
    "iris_x": "(iris.x - left.x) / eye_width",
    "iris_y": "(iris.y - (left.y + iris_x * (right.y - left.y))) / eye_width",
    "eye_open": "abs(lower.y - upper.y) / eye_width",
    "eyes": "MediaPipe normalized coordinates on horizontally mirrored frame. A: 33,133;159,145;468. B: 362,263;386,374;473.",
}
POSE_FORMULA = "MediaPipe facial matrix; SVD; D R D; RQ Euler in yaw,pitch,roll order"
REFERENCES = (
    ("prueba_ojos/headcontrol_gaze_model.joblib",
     "58a77a70385b82d272375106768cbfa2ba52fdb92b8adf1e916b3c3f8891ffeb"),
    ("prueba_ojos/gaze_pose_candidate_20261123_200853_033019/headcontrol_gaze_pose_candidate.joblib",
     "cb0cbdcc5af44326934f0f56d1b1110841c029883db265a0afca3eda948fc043"),
)


def verified(path, expected):
    contents = path.read_bytes()
    if sha256(contents).hexdigest() != expected:
        raise ValueError(f"SHA-256 inesperado: {path}")
    return contents


def features_for(name):
    return EYES if name in (MODELS[0], MODELS[2]) else EYES + POSE


def load_mlps():
    import torch

    sys.path.insert(0, str(TRAINING))
    from model import GazeMLP

    torch.set_num_threads(1)
    models, info = {}, {}
    for name in MODELS[:2]:
        features = features_for(name)
        width = len(features)
        meta_path = RUN3 / f"head_pose_{name}.json"
        meta_bytes = meta_path.read_bytes()
        meta = json.loads(meta_bytes)
        extraction, norm = meta["extraction"], meta["normalization"]
        if (meta["features"] != list(features) or meta["targets"] != list(TARGETS)
                or meta["architecture"] != [width, 64, 32, 16, 2] or meta["activation"] != "relu"
                or meta["feature_contract"] != f"features-v{1 if width == 6 else 2}"
                or meta["vision_contract_reference"] != "vision-v1"
                or meta["coordinate_contract"] != "coordinates-v1"
                or meta["dtype"] != "float32" or meta["output"] != "unclipped continuous [x, y]"
                or any(extraction.get(k) != v for k, v in EYE_FORMULAS.items())
                or extraction.get("ocular_units") != "dimensionless"
                or extraction.get("pose_units") != ("degrees" if width == 9 else None)
                or extraction.get("pose_formula") != (POSE_FORMULA if width == 9 else None)
                or norm["location"] != "model_buffers" or norm["method"] != "weighted_zscore"
                or norm.get("fitted_on") != "development_only"
                or len(norm["mean"]) != width or len(norm["std"]) != width):
            raise ValueError(f"Contrato MLP incompatible: {meta_path}")
        path = RUN3 / f"head_pose_{name}.pt"
        if path.name != meta["weights"]:
            raise ValueError(f"Archivo de pesos inesperado: {meta_path}")
        state = torch.load(BytesIO(verified(path, meta["weights_sha256"])), map_location="cpu", weights_only=True)
        model = GazeMLP(norm["mean"], norm["std"])
        if (not torch.equal(model.mean, state["mean"]) or not torch.equal(model.scale, state["scale"])
                or not all(torch.isfinite(value).all().item() for value in state.values())
                or not torch.all(model.scale > 0).item()):
            raise ValueError(f"Normalizador inválido: {path}")
        model.load_state_dict(state)
        models[name] = model.eval()
        info[name] = dict(path=path.relative_to(ROOT).as_posix(), sha256=meta["weights_sha256"],
                          metadata_sha256=sha256(meta_bytes).hexdigest(),
                          features=list(features), targets=list(TARGETS), normalization=norm,
                          provenance="phase3_head_pose_development")
    return torch, models, info


def serve_mlps():
    torch, models, info = load_mlps()
    print(json.dumps(info), flush=True)
    with torch.inference_mode():
        for line in sys.stdin:
            values = torch.tensor(json.loads(line), dtype=torch.float32)[None, :]
            output = {}
            for name, model in models.items():
                prediction = model(values[:, :len(features_for(name))])[0]
                output[name] = prediction.tolist() if torch.isfinite(prediction).all() else None
            print(json.dumps(output, allow_nan=False), flush=True)


def load_trees():
    import joblib

    models, info = {}, {}
    for name, (relative, expected) in zip(MODELS[2:], REFERENCES, strict=True):
        bundle = joblib.load(BytesIO(verified(ROOT / relative, expected)))
        features = features_for(name)
        model = bundle["model"]
        if (bundle["feature_names"] != list(features) or bundle["target_names"] != list(TARGETS)
                or model.n_features_in_ != len(features) or model.n_outputs_ != 2):
            raise ValueError(f"Contrato ExtraTrees incompatible: {relative}")
        if len(features) == 9 and bundle.get("pose_source") != (
            "MediaPipe facial transformation matrix, mirrored image, degrees"
        ):
            raise ValueError(f"Unidades de postura incompatibles: {relative}")
        models[name] = model
        info[name] = dict(path=relative, sha256=expected, features=list(features), targets=list(TARGETS),
                          normalization="none", provenance="archived_reference_not_phase3_refit")
    return models, info


@contextmanager
def prediction_engine():
    import joblib
    import numpy as np

    trees, reference_info = load_trees()
    python = TRAINING / ".venv" / "Scripts" / "python.exe"
    with subprocess.Popen([str(python), "-X", "utf8", "-B", str(__file__)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                          encoding="utf-8", bufsize=1) as worker:
        try:
            mlp_info = json.loads(worker.stdout.readline())
            manifest = {**mlp_info, **reference_info}

            def predict(vector):
                values = np.asarray(vector, dtype=np.float64)
                worker.stdin.write(json.dumps(values.tolist(), allow_nan=False) + "\n")
                worker.stdin.flush()
                result = json.loads(worker.stdout.readline())
                with joblib.parallel_backend("sequential"):
                    for name, model in trees.items():
                        prediction = model.predict(values[None, :len(features_for(name))])[0]
                        result[name] = prediction.tolist() if np.isfinite(prediction).all() else None
                return result

            yield manifest, predict
        finally:
            worker.stdin.close()


if __name__ == "__main__":
    serve_mlps()
