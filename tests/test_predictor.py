from __future__ import annotations

import numpy as np

from face.config import AppConfig
from face.predictor import FacePredictor
from face.predictor import _repair_flow_points


def test_predictor_detected_to_flow_to_predict_and_hide(monkeypatch):
    config = AppConfig(loss_ttl_ms=200, fade_out_ms=200)
    predictor = FacePredictor(config)

    base = np.array(
        [
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ],
        dtype=np.float32,
    )
    gray = np.zeros((64, 64), dtype=np.uint8)

    first = predictor.update(gray, base, timestamp_ms=1000)
    assert first.source == "detected"
    assert first.valid is True

    def fake_flow(_frame_gray):
        return base + 2.0

    monkeypatch.setattr(predictor, "_track_with_optical_flow", fake_flow)
    second = predictor.update(gray, None, timestamp_ms=1040)
    assert second.source == "flow"
    assert second.valid is True
    assert np.allclose(second.points, base + 2.0)

    monkeypatch.setattr(predictor, "_track_with_optical_flow", lambda _g: None)
    third = predictor.update(gray, None, timestamp_ms=1100)
    assert third.source == "predicted"
    assert third.alpha == 1.0
    assert third.valid is True

    fourth = predictor.update(gray, None, timestamp_ms=1260)
    assert fourth.source == "predicted"
    assert 0.0 < fourth.alpha < 1.0

    fifth = predictor.update(gray, None, timestamp_ms=1500)
    assert fifth.source == "hidden"
    assert fifth.valid is False
    assert fifth.points is None


def test_needs_redetect_interval():
    config = AppConfig(re_detect_interval=3)
    predictor = FacePredictor(config)
    assert predictor.needs_redetect(0) is True

    gray = np.zeros((32, 32), dtype=np.uint8)
    points = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    predictor.update(gray, points, timestamp_ms=1)

    assert predictor.needs_redetect(1) is False
    assert predictor.needs_redetect(2) is False
    assert predictor.needs_redetect(3) is True


def test_repair_flow_points_uses_affine_for_missing_points():
    prev = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    matrix = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    prev_h = np.hstack((prev, np.ones((prev.shape[0], 1), dtype=np.float32)))
    transformed = prev_h @ matrix.T

    next_points = transformed.copy()
    next_points[3] = np.array([999.0, 999.0], dtype=np.float32)
    status = np.array([True, True, True, False], dtype=bool)

    class FakeCV2:
        RANSAC = 1

        @staticmethod
        def estimateAffinePartial2D(*_args, **_kwargs):
            return matrix, np.ones((3, 1), dtype=np.uint8)

    repaired = _repair_flow_points(prev, next_points, status, FakeCV2)

    assert np.allclose(repaired[3], transformed[3], atol=1e-6)
