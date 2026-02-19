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

        mask = _build_polygon_mask(height, width, polygon, self._cv2)
        if not np.any(mask):
            return frame

        x_min = int(np.min(polygon[:, 0]))
        x_max = int(np.max(polygon[:, 0])) + 1
        y_min = int(np.min(polygon[:, 1]))
        y_max = int(np.max(polygon[:, 1])) + 1
        x_min = int(np.clip(x_min, 0, width))
        x_max = int(np.clip(x_max, 0, width))
        y_min = int(np.clip(y_min, 0, height))
        y_max = int(np.clip(y_max, 0, height))
        if x_min >= x_max or y_min >= y_max:
            return frame

        overlay = frame.copy()
        roi = frame[y_min:y_max, x_min:x_max]
        roi_mask = mask[y_min:y_max, x_min:x_max]
        pixelated_roi = _pixelate_roi(roi, self.config.face_pixel_size, self._cv2)
        overlay_roi = overlay[y_min:y_max, x_min:x_max]
        overlay_roi[roi_mask] = pixelated_roi[roi_mask]

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


def _build_polygon_mask(height: int, width: int, polygon: np.ndarray, cv2_module) -> np.ndarray:
    if cv2_module is not None:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2_module.fillPoly(mask, [polygon], 255)
        return mask > 0
    return _polygon_mask(height, width, polygon.astype(np.float32))


def _pixelate_roi(roi: np.ndarray, block_size: int, cv2_module) -> np.ndarray:
    if roi.size == 0:
        return roi

    block = max(1, int(block_size))
    roi_h, roi_w = roi.shape[:2]
    small_w = max(1, roi_w // block)
    small_h = max(1, roi_h // block)

    if cv2_module is not None:
        reduced = cv2_module.resize(
            roi, (small_w, small_h), interpolation=cv2_module.INTER_AREA
        )
        return cv2_module.resize(
            reduced, (roi_w, roi_h), interpolation=cv2_module.INTER_NEAREST
        )

    reduced = roi[::block, ::block]
    if reduced.shape[0] == 0 or reduced.shape[1] == 0:
        reduced = roi[:1, :1]
    expanded = np.repeat(np.repeat(reduced, block, axis=0), block, axis=1)
    return expanded[:roi_h, :roi_w]


def _scale_polygon(
    polygon: np.ndarray, scale: float, width: int, height: int
) -> np.ndarray:
    center = np.mean(polygon.astype(np.float32), axis=0, keepdims=True)
    expanded = center + (polygon.astype(np.float32) - center) * float(scale)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return np.round(expanded).astype(np.int32)
