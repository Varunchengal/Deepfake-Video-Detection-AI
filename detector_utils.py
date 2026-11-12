# detector_utils.py - YuNet face extraction helpers for OpenCV 5.x

from pathlib import Path

import cv2
import numpy as np


_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

_FACE_DETECTOR = None
_FACE_DETECTOR_LOAD_ATTEMPTED = False


def _create_yunet_detector():
    """
    Create an OpenCV YuNet detector.

    Required model:
        models/face_detection_yunet_2023mar.onnx

    Returns:
        A FaceDetectorYN instance, or None if the model/API is unavailable.
    """
    if not _MODEL_PATH.is_file():
        return None

    try:
        # OpenCV builds expose either FaceDetectorYN.create or
        # the convenience function FaceDetectorYN_create.
        face_detector_class = getattr(cv2, "FaceDetectorYN", None)
        if face_detector_class is not None and hasattr(face_detector_class, "create"):
            return face_detector_class.create(
                str(_MODEL_PATH),
                "",
                (320, 320),
                0.75,   # score threshold
                0.30,   # NMS threshold
                5000,   # top_k
            )

        create_function = getattr(cv2, "FaceDetectorYN_create", None)
        if create_function is not None:
            return create_function(
                str(_MODEL_PATH),
                "",
                (320, 320),
                0.75,
                0.30,
                5000,
            )
    except (cv2.error, AttributeError, TypeError, RuntimeError):
        return None

    return None


def _get_face_detector():
    """Load YuNet once and reuse it for subsequent frames."""
    global _FACE_DETECTOR, _FACE_DETECTOR_LOAD_ATTEMPTED

    if not _FACE_DETECTOR_LOAD_ATTEMPTED:
        _FACE_DETECTOR = _create_yunet_detector()
        _FACE_DETECTOR_LOAD_ATTEMPTED = True

    return _FACE_DETECTOR


def face_detector_available():
    """Return True only when the YuNet model and OpenCV API are available."""
    return _get_face_detector() is not None


def extract_face_region(frame, padding=0.20):
    """
    Detect the largest face in an OpenCV BGR frame using YuNet.

    Returns an RGB uint8 face crop, or None if no face is detected.
    There is deliberately no centre-crop fallback because the training
    dataset contains face-only crops.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Received an empty or invalid video frame.")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR frame from OpenCV.")

    detector = _get_face_detector()
    if detector is None:
        raise RuntimeError(
            "YuNet detector is unavailable. Check models/face_detection_yunet_2023mar.onnx."
        )

    height, width = frame.shape[:2]
    if width < 20 or height < 20:
        return None

    try:
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)
    except (cv2.error, AttributeError, TypeError, RuntimeError):
        return None

    if faces is None or len(faces) == 0:
        return None

    # YuNet rows: x, y, width, height, landmarks..., confidence.
    # Pick the largest face; if areas tie, prefer the higher confidence.
    face = max(
        faces,
        key=lambda row: (
            max(0.0, float(row[2])) * max(0.0, float(row[3])),
            float(row[14]) if len(row) > 14 else 0.0,
        ),
    )

    x, y, face_width, face_height = map(float, face[:4])
    if face_width <= 0 or face_height <= 0:
        return None

    pad_x = int(face_width * max(0.0, float(padding)))
    pad_y = int(face_height * max(0.0, float(padding)))

    x1 = max(0, int(np.floor(x)) - pad_x)
    y1 = max(0, int(np.floor(y)) - pad_y)
    x2 = min(width, int(np.ceil(x + face_width)) + pad_x)
    y2 = min(height, int(np.ceil(y + face_height)) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    face_bgr = frame[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return None

    return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)


def analyze_texture_artifacts(image_np):
    """
    Return Laplacian variance as a basic sharpness statistic.
    This is not a deepfake score and must not adjust model probabilities.
    """
    if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
        raise ValueError("Received an empty or invalid image.")

    if image_np.ndim == 2:
        gray = image_np
    elif image_np.ndim == 3 and image_np.shape[2] == 3:
        # extract_face_region() returns RGB.
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Expected a grayscale or 3-channel RGB image.")

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
