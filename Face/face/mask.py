from __future__ import annotations

import numpy as np

from .config import AppConfig
from .predictor import PredictedTrack


class MaskRenderer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError:
            cv2 = None
        self._cv2 = cv2

    def draw(self, frame: np.ndarray, track: PredictedTrack) -> np.ndarray:
        if track.points is None or track.alpha <= 0.0:
            return frame
        if track.points.shape[0] < 3:
            return frame

        height, width = frame.shape[:2]
        polygon = np.round(track.points).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

        if self._cv2 is not None:
            hull = self._cv2.convexHull(polygon.reshape(-1, 1, 2)).reshape(-1, 2)
        else:
            hull = polygon
        polygon = _scale_polygon(hull, self.config.mask_scale, width, height)

        alpha = float(np.clip(self.config.mask_alpha * track.alpha, 0.0, 1.0))
        if alpha <= 0.0:
            return frame

        overlay = frame.copy()
        if self._cv2 is not None:
            self._cv2.fillPoly(overlay, [polygon], self.config.mask_color_bgr)
            _draw_cartoon_eyes_cv2(self._cv2, overlay, polygon)
        else:
            mask = _polygon_mask(height, width, polygon.astype(np.float32))
            overlay[mask] = np.array(self.config.mask_color_bgr, dtype=np.uint8)

        if alpha >= 1.0:
            return overlay
        return _alpha_blend(overlay, frame, alpha)


def _alpha_blend(overlay: np.ndarray, frame: np.ndarray, alpha: float) -> np.ndarray:
    blended = overlay.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _polygon_mask(height: int, width: int, polygon: np.ndarray) -> np.ndarray:
    # Ray casting fallback when OpenCV is unavailable (used primarily in tests).
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx + 0.5
    y = yy + 0.5
    inside = np.zeros((height, width), dtype=bool)

    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        intersects = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-9) + x1
        )
        inside ^= intersects
    return inside


def _scale_polygon(
    polygon: np.ndarray, scale: float, width: int, height: int
) -> np.ndarray:
    center = np.mean(polygon.astype(np.float32), axis=0, keepdims=True)
    expanded = center + (polygon.astype(np.float32) - center) * float(scale)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return np.round(expanded).astype(np.int32)


def _draw_cartoon_eyes_cv2(cv2_module, canvas: np.ndarray, polygon: np.ndarray) -> None:
    x_min = int(np.min(polygon[:, 0]))
    x_max = int(np.max(polygon[:, 0]))
    y_min = int(np.min(polygon[:, 1]))
    y_max = int(np.max(polygon[:, 1]))
    face_w = max(1, x_max - x_min)
    face_h = max(1, y_max - y_min)
    center_x = int((x_min + x_max) * 0.5)

    eye_y = int(y_min + 0.4 * face_h)
    eye_dx = max(10, int(face_w * 0.19))
    eye_rx = max(7, int(face_w * 0.078))
    eye_ry = max(4, int(face_h * 0.05))
    pupil_r = max(2, int(min(face_w, face_h) * 0.016))

    left_eye = (center_x - eye_dx, eye_y)
    right_eye = (center_x + eye_dx, eye_y)
    height, width = canvas.shape[:2]
    left_eye = (int(np.clip(left_eye[0], 0, width - 1)), int(np.clip(left_eye[1], 0, height - 1)))
    right_eye = (
        int(np.clip(right_eye[0], 0, width - 1)),
        int(np.clip(right_eye[1], 0, height - 1)),
    )

    sclera_color = (230, 245, 230)
    lid_color = (30, 95, 30)
    pupil_color = (20, 45, 20)

    # Neutral eyes with soft colors.
    cv2_module.ellipse(canvas, left_eye, (eye_rx, eye_ry), 0, 0, 360, sclera_color, -1)
    cv2_module.ellipse(canvas, right_eye, (eye_rx, eye_ry), 0, 0, 360, sclera_color, -1)
    cv2_module.ellipse(canvas, left_eye, (eye_rx, eye_ry), 0, 0, 360, lid_color, 1)
    cv2_module.ellipse(canvas, right_eye, (eye_rx, eye_ry), 0, 0, 360, lid_color, 1)

    # Small soft pupils.
    cv2_module.circle(canvas, left_eye, pupil_r, pupil_color, -1)
    cv2_module.circle(canvas, right_eye, pupil_r, pupil_color, -1)

    # Very light upper lids to keep a neutral expression.
    lid_thickness = 1
    cv2_module.line(
        canvas,
        (left_eye[0] - eye_rx, left_eye[1] - eye_ry),
        (left_eye[0] + eye_rx, left_eye[1] - eye_ry),
        lid_color,
        lid_thickness,
        cv2_module.LINE_AA,
    )
    cv2_module.line(
        canvas,
        (right_eye[0] - eye_rx, right_eye[1] - eye_ry),
        (right_eye[0] + eye_rx, right_eye[1] - eye_ry),
        lid_color,
        lid_thickness,
        cv2_module.LINE_AA,
    )
