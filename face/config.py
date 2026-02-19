from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def parse_bgr_color(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Mask color must contain exactly 3 comma-separated values.")
    channels = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("Mask color channels must be in range 0..255.")
    return channels


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    show_fps: bool = False
    flip_horizontal: bool = True
    mask_color_bgr: tuple[int, int, int] = (0, 255, 0)
    mask_alpha: float = 1.0
    mask_scale: float = 1.18
    face_pixel_size: int = 16
    loss_ttl_ms: int = 200
    fade_out_ms: int = 250
    re_detect_interval: int = 2
    flow_min_ratio: float = 0.55
    flow_win_size: int = 21
    flow_max_level: int = 3
    flow_max_iters: int = 20
    flow_eps: float = 0.03
    model_path: str = "models/face_landmarker.task"
    enable_background_removal: bool = True
    background_color_bgr: tuple[int, int, int] = (255, 255, 0)
    background_blur_kernel: int = 61
    background_threshold: float = 0.2
    background_smoothing: float = 0.65
    background_hysteresis: float = 0.08
    background_inference_interval: int = 2
    background_input_scale: float = 0.6
    background_model_path: str = "models/selfie_segmenter.tflite"
    enable_hand_tracking: bool = True
    hand_model_path: str = "models/hand_landmarker.task"
    hand_mask_scale: float = 1.35
    hand_mask_dilate: int = 2
    hand_inference_interval: int = 2
    hand_input_scale: float = 0.65
    enable_virtual_cam: bool = False
    virtual_cam_device: str = "/dev/video10"
    virtual_cam_backend: str = "v4l2loopback"
    virtual_cam_fps: float = 30.0
    show_preview: bool = True

    @classmethod
    def from_args(cls, args: Any) -> "AppConfig":
        alpha = max(0.0, min(1.0, float(args.alpha)))
        return cls(
            camera_index=int(args.camera),
            frame_width=int(args.width),
            frame_height=int(args.height),
            show_fps=bool(args.show_fps),
            flip_horizontal=not bool(args.no_flip),
            mask_color_bgr=parse_bgr_color(args.mask_color),
            mask_alpha=alpha,
            mask_scale=max(1.0, min(1.8, float(args.mask_scale))),
            face_pixel_size=max(4, min(24, int(args.face_pixel_size))),
            loss_ttl_ms=max(0, int(args.loss_ttl_ms)),
            fade_out_ms=max(1, int(args.fade_out_ms)),
            re_detect_interval=max(1, int(args.re_detect_interval)),
            flow_min_ratio=max(0.0, min(1.0, float(args.flow_min_ratio))),
            model_path=str(args.model_path),
            enable_background_removal=not bool(args.disable_bg_remove),
            background_color_bgr=parse_bgr_color(args.bg_color),
            background_blur_kernel=_normalize_odd_kernel(int(args.bg_blur_kernel)),
            background_threshold=max(0.0, min(0.99, float(args.bg_threshold))),
            background_smoothing=max(0.0, min(1.0, float(args.bg_smoothing))),
            background_hysteresis=max(0.0, min(0.35, float(args.bg_hysteresis))),
            background_inference_interval=max(1, int(args.bg_interval)),
            background_input_scale=max(0.3, min(1.0, float(args.bg_scale))),
            background_model_path=str(args.bg_model_path),
            enable_hand_tracking=not bool(args.disable_hands),
            hand_model_path=str(args.hand_model_path),
            hand_mask_scale=max(1.0, min(2.5, float(args.hand_mask_scale))),
            hand_mask_dilate=max(0, min(8, int(args.hand_mask_dilate))),
            hand_inference_interval=max(1, int(args.hand_interval)),
            hand_input_scale=max(0.35, min(1.0, float(args.hand_scale))),
            enable_virtual_cam=bool(args.virtual_cam),
            virtual_cam_device=str(args.virtual_cam_device),
            virtual_cam_backend=str(args.virtual_cam_backend),
            virtual_cam_fps=max(1.0, min(120.0, float(args.virtual_cam_fps))),
            show_preview=not bool(args.no_preview),
        )


def _normalize_odd_kernel(value: int) -> int:
    value = max(3, min(151, value))
    if value % 2 == 0:
        value += 1
    return value
