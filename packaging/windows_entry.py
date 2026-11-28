"""Launch the Windows UI using the same relative resources from source or the portable build."""

import argparse
from pathlib import Path
import sys


def run():
    parser = argparse.ArgumentParser(description="HeadControl AI")
    parser.add_argument("--self-test", action="store_true", help="Comprueba los recursos locales sin cámara.")
    args = parser.parse_args()
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(root / "product/mvp"))
    from tracker_windows import main

    if not args.self_test:
        return main()

    from contextlib import ExitStack
    from hashlib import sha256
    import cv2
    import mediapipe as mp
    import numpy as np
    from calibrate_user import GUIDED_RUNTIME, GUIDED_V2_RUNTIME, RUNTIME, Tracker
    from features import FACE_TASK_PATH, FACE_TASK_HASH

    for runtime in (RUNTIME, GUIDED_RUNTIME, GUIDED_V2_RUNTIME):
        tracker = Tracker(runtime)
        result = tracker.session.run(["coordinates"], {"features": np.zeros((1, 9), np.float32)})[0]
        if result.shape != (1, 2) or not np.isfinite(result).all():
            raise RuntimeError("Salida ONNX inválida.")
    if sha256(FACE_TASK_PATH.read_bytes()).hexdigest() != FACE_TASK_HASH:
        raise RuntimeError("Huella de Face Landmarker inválida.")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)))
    with ExitStack() as stack:
        detector = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(options))
        rgb = cv2.cvtColor(np.zeros((160, 160, 3), np.uint8), cv2.COLOR_BGR2RGB)
        detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    from PySide6.QtWidgets import QApplication
    from tracker_windows import Window
    app = QApplication([])
    window = Window(consensus=True)
    window.timer.stop()
    window.model.setCurrentIndex(2)
    if window.armed or window.vision.isRunning() or not window.fixation_active():
        raise RuntimeError("Estado de arranque inesperado.")
    window.close()
    print("OK: GUI, tres ONNX, normalizador, OpenCV y Face Landmarker; cámara y cursor desactivados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
