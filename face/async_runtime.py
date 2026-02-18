from __future__ import annotations

import threading
import time

import numpy as np

from .background import BackgroundRemover
from .hands import HandTracker


def _clone_polygons(polygons: list[np.ndarray]) -> list[np.ndarray]:
    return [polygon.copy() for polygon in polygons]


class AsyncHandRunner:
    def __init__(self, tracker: HandTracker) -> None:
        self._tracker = tracker
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: tuple[np.ndarray, int] | None = None
        self._latest: list[np.ndarray] = []
        self._thread = threading.Thread(target=self._loop, daemon=True, name="hand-runner")
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        with self._lock:
            if self._pending is not None:
                return
            self._pending = (frame_bgr.copy(), int(timestamp_ms))

    def latest(self) -> list[np.ndarray]:
        with self._lock:
            return _clone_polygons(self._latest)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._tracker.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job: tuple[np.ndarray, int] | None = None
            with self._lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
            if job is None:
                time.sleep(0.001)
                continue

            frame_bgr, timestamp_ms = job
            try:
                result = self._tracker.detect(frame_bgr, timestamp_ms)
            except Exception:
                result = []
            with self._lock:
                self._latest = _clone_polygons(result)


class AsyncBackgroundRunner:
    def __init__(self, remover: BackgroundRemover) -> None:
        self._remover = remover
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pending: tuple[np.ndarray, int, list[np.ndarray]] | None = None
        self._latest_alpha: np.ndarray | None = None
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="background-runner"
        )
        self._thread.start()

    def submit(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        hand_polygons: list[np.ndarray],
    ) -> None:
        with self._lock:
            if self._pending is not None:
                return
            self._pending = (frame_bgr.copy(), int(timestamp_ms), _clone_polygons(hand_polygons))

    def render(self, frame_bgr: np.ndarray, hand_polygons: list[np.ndarray]) -> np.ndarray:
        with self._lock:
            alpha = None if self._latest_alpha is None else self._latest_alpha.copy()
        if alpha is None:
            return frame_bgr
        return self._remover.compose_with_alpha(frame_bgr, alpha, hand_polygons)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._remover.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            job: tuple[np.ndarray, int, list[np.ndarray]] | None = None
            with self._lock:
                if self._pending is not None:
                    job = self._pending
                    self._pending = None
            if job is None:
                time.sleep(0.001)
                continue

            frame_bgr, timestamp_ms, hand_polygons = job
            try:
                alpha = self._remover.estimate_alpha(frame_bgr, timestamp_ms, hand_polygons)
            except Exception:
                alpha = None
            if alpha is None:
                continue
            with self._lock:
                self._latest_alpha = alpha.copy()
