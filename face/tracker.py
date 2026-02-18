from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class TrackResult:
    points: np.ndarray | None
    confidence: float
    timestamp_ms: int


class FaceMeshTracker:
    def __init__(self, model_path: str) -> None:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import mediapipe as mp  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import BaseOptions  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import vision  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for tracking. Install opencv-python and mediapipe."
            ) from exc

        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.exists():
            raise RuntimeError(
                f"FaceLandmarker model not found: {resolved_model}\n"
                "Download it with:\n"
                "curl -L "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
                f"-o {resolved_model}"
            )

        self._cv2 = cv2
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._face_oval_indices = self._build_face_oval_indices(
            vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
        )

    @staticmethod
    def _build_face_oval_indices(connections) -> list[int]:
        seen = set()
        ordered: list[int] = []
        for connection in connections:
            start = int(connection.start)
            end = int(connection.end)
            if start not in seen:
                ordered.append(start)
                seen.add(start)
            if end not in seen:
                ordered.append(end)
                seen.add(end)
        return ordered

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> TrackResult:
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return TrackResult(points=None, confidence=0.0, timestamp_ms=timestamp_ms)

        face_landmarks = result.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        oval_points = np.array(
            [
                (
                    float(face_landmarks[index].x * width),
                    float(face_landmarks[index].y * height),
                )
                for index in self._face_oval_indices
            ],
            dtype=np.float32,
        )
        return TrackResult(points=oval_points, confidence=1.0, timestamp_ms=timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()
