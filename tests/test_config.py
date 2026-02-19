from __future__ import annotations

from face.config import AppConfig
from main import build_parser


def test_face_pixel_size_clamps_max_and_mask_color_is_compatible():
    parser = build_parser()
    args = parser.parse_args(["--face-pixel-size", "30", "--mask-color", "1,2,3"])
    config = AppConfig.from_args(args)

    assert config.face_pixel_size == 24
    assert config.mask_color_bgr == (1, 2, 3)


def test_face_pixel_size_clamps_min():
    parser = build_parser()
    args = parser.parse_args(["--face-pixel-size", "2"])
    config = AppConfig.from_args(args)

    assert config.face_pixel_size == 4
