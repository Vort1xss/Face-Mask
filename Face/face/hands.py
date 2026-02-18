from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AppConfig


class HandTracker:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import mediapipe as mp  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import BaseOptions  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import vision  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for hand tracking. Install opencv-python and mediapipe."
            ) from exc

        resolved_model = Path(config.hand_model_path).expanduser().resolve()
        if not resolved_model.exists():
            raise RuntimeError(
                f"Hand landmarker model not found: {resolved_model}\n"
                "Download it with:\n"
                "curl -L "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task "
                f"-o {resolved_model}"
            )

        self._cv2 = cv2
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        )
        self._frame_index = 0
        self._cached_hands: list[np.ndarray] | None = None

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[np.ndarray]:
        self._frame_index += 1
        if (
            self._cached_hands is not None
            and self._frame_index % self._config.hand_inference_interval != 0
        ):
            return self._cached_hands

        frame_for_model = frame_bgr
        scale = self._config.hand_input_scale
        inv_scale = 1.0
        if scale < 0.999:
            height, width = frame_bgr.shape[:2]
            model_w = max(64, int(width * scale))
            model_h = max(64, int(height * scale))
            frame_for_model = self._cv2.resize(
                frame_bgr, (model_w, model_h), interpolation=self._cv2.INTER_LINEAR
            )
            inv_scale = 1.0 / scale

        frame_rgb = self._cv2.cvtColor(frame_for_model, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._cached_hands = []
            return self._cached_hands

        height, width = frame_for_model.shape[:2]
        hands: list[np.ndarray] = []
        for landmarks in result.hand_landmarks:
            points = np.array(
                [(float(landmark.x * width), float(landmark.y * height)) for landmark in landmarks],
                dtype=np.float32,
            )
            if inv_scale != 1.0:
                points *= inv_scale
            hands.append(points)
        self._cached_hands = hands
        return hands

    def close(self) -> None:
        self._landmarker.close()
