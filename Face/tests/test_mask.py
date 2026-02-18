from __future__ import annotations

import numpy as np

from face.config import AppConfig
from face.mask import MaskRenderer
from face.predictor import PredictedTrack


def test_mask_renderer_fills_polygon():
    config = AppConfig(mask_color_bgr=(0, 255, 0), mask_alpha=1.0)
    renderer = MaskRenderer(config)

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    points = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    track = PredictedTrack(points=points, source="detected", age_ms=0, valid=True, alpha=1.0)

    out = renderer.draw(frame, track)

    assert tuple(out[20, 20]) == (0, 255, 0)
    assert tuple(out[5, 5]) == (0, 0, 0)

