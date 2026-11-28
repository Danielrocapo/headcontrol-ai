"""Añade las salidas oculares al detector Windows sin alterar Camera9."""

from features import FACE_TASK_PATH, mp
from wink_clicks import eye_measurement


class EyeOutputs:
    def __init__(self, detector):
        self.detector = detector
        self.categories = ()

    def detect_for_video(self, image, timestamp):
        result = self.detector.detect_for_video(image, timestamp)
        self.categories = result.face_blendshapes[0] if result.face_blendshapes else ()
        return result


def enable_eye_outputs(camera, stack):
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_TASK_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=camera.metadata["num_faces"],
        min_face_detection_confidence=camera.metadata["thresholds"]["face_detection"],
        min_face_presence_confidence=camera.metadata["thresholds"]["face_presence"],
        min_tracking_confidence=camera.metadata["thresholds"]["tracking"],
        output_facial_transformation_matrixes=True, output_face_blendshapes=True,
    )
    try:
        detector = stack.enter_context(mp.tasks.vision.FaceLandmarker.create_from_options(options))
    except (RuntimeError, ValueError):
        return
    # El cierre de FaceLandmarker es idempotente; ExitStack conserva ambos cierres.
    camera.detector.close()
    camera.detector = EyeOutputs(detector)


def read_eyes(camera, row, landmarks):
    if not row["valid_detection"]:
        return None
    return eye_measurement(getattr(camera.detector, "categories", ()), landmarks,
                           row["camera_width"], row["camera_height"], camera.metadata["mirrored"])
