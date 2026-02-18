from __future__ import annotations

import numpy as np
import cv2

from face.background import composite_background
from face.background import compute_stable_binary_mask
from face.background import force_foreground_by_polygons


def test_compute_stable_binary_mask_with_hysteresis_keeps_prev():
    confidence = np.array([[0.20, 0.23, 0.40]], dtype=np.float32)
    prev = np.array([[1, 0, 0]], dtype=np.uint8)
    binary = compute_stable_binary_mask(
        confidence_mask=confidence,
        threshold=0.25,
        hysteresis=0.08,
        prev_binary=prev,
    )
    assert np.array_equal(binary, np.array([[1, 0, 1]], dtype=np.uint8))


def test_composite_background_respects_alpha():
    frame = np.array(
        [[[10, 20, 30], [40, 50, 60]]],
        dtype=np.uint8,
    )
    blue_bg = (255, 255, 0)
    alpha = np.array([[1.0, 0.0]], dtype=np.float32)
    out = composite_background(frame, blue_bg, alpha)

    assert tuple(out[0, 0]) == (10, 20, 30)
    assert tuple(out[0, 1]) == blue_bg


def test_force_foreground_by_polygons_marks_hand_region():
    base = np.zeros((20, 20), dtype=np.uint8)
    hand = np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.float32)
    out = force_foreground_by_polygons(
        binary_mask=base,
        polygons=[hand],
        width=20,
        height=20,
        cv2_module=cv2,
        scale=1.0,
        dilate_iters=0,
    )
    assert out[10, 10] == 1
