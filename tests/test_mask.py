from __future__ import annotations

import numpy as np

from face.config import AppConfig
from face.mask import MaskRenderer
from face.predictor import PredictedTrack


def test_mask_renderer_pixelates_face_polygon():
    config = AppConfig(mask_alpha=1.0, mask_scale=1.0, face_pixel_size=8)
    renderer = MaskRenderer(config)

    yy, xx = np.mgrid[0:40, 0:40]
    frame = np.stack(
        (
            (xx * 3) % 256,
            (yy * 5) % 256,
            ((xx + yy) * 7) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    points = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)
    track = PredictedTrack(points=points, source="detected", age_ms=0, valid=True, alpha=1.0)

    out = renderer.draw(frame, track)

    assert np.array_equal(out[:8, :8], frame[:8, :8])
    inside_before = frame[10:31, 10:31]
    inside_after = out[10:31, 10:31]
    changed_pixels = np.count_nonzero(np.any(inside_after != inside_before, axis=2))
    assert changed_pixels > 0

    before_unique = np.unique(frame[12:20, 12:20].reshape(-1, 3), axis=0).shape[0]
    after_unique = np.unique(out[12:20, 12:20].reshape(-1, 3), axis=0).shape[0]
    assert after_unique < before_unique
