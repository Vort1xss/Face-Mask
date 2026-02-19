from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AppConfig


@dataclass(slots=True)
class PredictedTrack:
    points: np.ndarray | None
    source: str
    age_ms: int
    valid: bool
    alpha: float


class FacePredictor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.last_valid_points: np.ndarray | None = None
        self.last_valid_time_ms: int | None = None
        self.last_velocity: np.ndarray | None = None

        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError:
            cv2 = None
        self._cv2 = cv2

    def needs_redetect(self, frame_index: int) -> bool:
        if self.last_valid_points is None:
            return True
        return frame_index % self.config.re_detect_interval == 0

    def update(
        self,
        frame_gray: np.ndarray,
        detected_points: np.ndarray | None,
        timestamp_ms: int,
    ) -> PredictedTrack:
        if detected_points is not None:
            points = detected_points.astype(np.float32, copy=True)
            self._commit_valid(frame_gray, points, timestamp_ms)
            return PredictedTrack(
                points=points, source="detected", age_ms=0, valid=True, alpha=1.0
            )

        flow_points = self._track_with_optical_flow(frame_gray)
        if flow_points is not None:
            age = self._age_since_valid(timestamp_ms)
            self._commit_valid(frame_gray, flow_points, timestamp_ms)
            return PredictedTrack(
                points=flow_points, source="flow", age_ms=age, valid=True, alpha=1.0
            )

        predicted = self._predict_without_observation(timestamp_ms)
        self.prev_gray = frame_gray.copy()
        if predicted is None:
            return PredictedTrack(
                points=None,
                source="hidden",
                age_ms=self._age_since_valid(timestamp_ms),
                valid=False,
                alpha=0.0,
            )
        points, alpha, age_ms = predicted
        return PredictedTrack(
            points=points, source="predicted", age_ms=age_ms, valid=alpha > 0.0, alpha=alpha
        )

    def _track_with_optical_flow(self, frame_gray: np.ndarray) -> np.ndarray | None:
        if self._cv2 is None or self.prev_gray is None or self.prev_points is None:
            return None

        criteria = (
            self._cv2.TERM_CRITERIA_EPS | self._cv2.TERM_CRITERIA_COUNT,
            self.config.flow_max_iters,
            self.config.flow_eps,
        )
        next_points, status, _ = self._cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            self.prev_points,
            None,
            winSize=(self.config.flow_win_size, self.config.flow_win_size),
            maxLevel=self.config.flow_max_level,
            criteria=criteria,
        )
        if next_points is None or status is None:
            return None

        status_flat = status.reshape(-1).astype(bool)
        good_ratio = float(np.mean(status_flat))
        if good_ratio < self.config.flow_min_ratio:
            return None

        prev_flat = self.prev_points.reshape(-1, 2)
        next_flat = next_points.reshape(-1, 2)
        repaired = _repair_flow_points(prev_flat, next_flat, status_flat, self._cv2)
        return repaired.astype(np.float32)

    def _predict_without_observation(
        self, timestamp_ms: int
    ) -> tuple[np.ndarray, float, int] | None:
        if self.last_valid_points is None or self.last_valid_time_ms is None:
            return None

        age_ms = max(0, timestamp_ms - self.last_valid_time_ms)
        velocity = self.last_velocity
        if velocity is None:
            velocity = np.zeros_like(self.last_valid_points)

        if age_ms <= self.config.loss_ttl_ms:
            predicted = self.last_valid_points + velocity * age_ms
            return predicted.astype(np.float32), 1.0, age_ms

        fade_age_ms = age_ms - self.config.loss_ttl_ms
        alpha = max(0.0, 1.0 - (fade_age_ms / self.config.fade_out_ms))
        if alpha <= 0.0:
            return None

        predicted = self.last_valid_points + velocity * self.config.loss_ttl_ms
        return predicted.astype(np.float32), alpha, age_ms

    def _commit_valid(
        self, frame_gray: np.ndarray, points: np.ndarray, timestamp_ms: int
    ) -> None:
        if self.last_valid_points is not None and self.last_valid_time_ms is not None:
            delta_ms = max(1, timestamp_ms - self.last_valid_time_ms)
            self.last_velocity = (points - self.last_valid_points) / float(delta_ms)
        else:
            self.last_velocity = np.zeros_like(points)

        self.last_valid_points = points.copy()
        self.last_valid_time_ms = timestamp_ms
        self.prev_points = points.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = frame_gray.copy()

    def _age_since_valid(self, timestamp_ms: int) -> int:
        if self.last_valid_time_ms is None:
            return 0
        return max(0, timestamp_ms - self.last_valid_time_ms)


def _repair_flow_points(
    prev_flat: np.ndarray, next_flat: np.ndarray, status_flat: np.ndarray, cv2_module
) -> np.ndarray:
    good_prev = prev_flat[status_flat]
    good_next = next_flat[status_flat]
    if good_prev.size == 0:
        return next_flat.copy()

    repaired = next_flat.copy()
    if cv2_module is not None and good_prev.shape[0] >= 3:
        matrix, _ = cv2_module.estimateAffinePartial2D(
            good_prev.astype(np.float32),
            good_next.astype(np.float32),
            method=cv2_module.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is not None:
            prev_h = np.hstack(
                (prev_flat.astype(np.float32), np.ones((prev_flat.shape[0], 1), dtype=np.float32))
            )
            projected = (prev_h @ matrix.T).astype(np.float32)
            repaired[~status_flat] = projected[~status_flat]
            return repaired

    median_delta = np.median(good_next - good_prev, axis=0)
    repaired[~status_flat] = prev_flat[~status_flat] + median_delta
    return repaired
