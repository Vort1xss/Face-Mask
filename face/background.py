from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AppConfig


def compute_stable_binary_mask(
    confidence_mask: np.ndarray,
    threshold: float,
    hysteresis: float,
    prev_binary: np.ndarray | None,
) -> np.ndarray:
    mask = np.squeeze(confidence_mask).astype(np.float32)
    high = min(1.0, threshold + hysteresis)
    low = max(0.0, threshold - hysteresis)

    binary = mask >= high
    if prev_binary is not None:
        keep_prev = prev_binary.astype(bool) & (mask >= low)
        binary = binary | keep_prev
    return binary.astype(np.uint8)


def force_foreground_by_polygons(
    binary_mask: np.ndarray,
    polygons: list[np.ndarray],
    width: int,
    height: int,
    cv2_module,
    scale: float,
    dilate_iters: int,
) -> np.ndarray:
    if not polygons:
        return binary_mask

    hand_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if polygon is None or polygon.shape[0] < 3:
            continue

        points = np.round(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        hull = cv2_module.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
        scaled = _scale_polygon(hull, scale, width, height)
        cv2_module.fillPoly(hand_mask, [scaled], 255)

    if dilate_iters > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        hand_mask = cv2_module.dilate(hand_mask, kernel, iterations=dilate_iters)

    forced = binary_mask.copy()
    forced[hand_mask > 0] = 1
    return forced


def composite_background(
    frame_bgr: np.ndarray, background_color_bgr: tuple[int, int, int], alpha: np.ndarray
) -> np.ndarray:
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    fg = frame_bgr.astype(np.float32)
    bg = np.full_like(frame_bgr, background_color_bgr, dtype=np.uint8).astype(np.float32)
    output = fg * alpha + bg * (1.0 - alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def composite_blur_background(
    frame_bgr: np.ndarray, alpha: np.ndarray, cv2_module, blur_kernel: int
) -> np.ndarray:
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blurred = cv2_module.GaussianBlur(frame_bgr, (blur_kernel, blur_kernel), 0)
    fg = frame_bgr.astype(np.float32)
    bg = blurred.astype(np.float32)
    output = fg * alpha + bg * (1.0 - alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


class BackgroundRemover:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import mediapipe as mp  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import BaseOptions  # pylint: disable=import-outside-toplevel
            from mediapipe.tasks.python import vision  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for background removal. Install opencv-python and mediapipe."
            ) from exc

        resolved_model = Path(config.background_model_path).expanduser().resolve()
        if not resolved_model.exists():
            raise RuntimeError(
                f"Selfie segmenter model not found: {resolved_model}\n"
                "Download it with:\n"
                "curl -L "
                "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite "
                f"-o {resolved_model}"
            )

        self._cv2 = cv2
        self._mp = mp
        self._segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_model)),
                running_mode=vision.RunningMode.VIDEO,
                output_confidence_masks=True,
                output_category_mask=False,
            )
        )
        self._prev_binary: np.ndarray | None = None
        self._frame_index = 0

    def estimate_alpha(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        hand_polygons: list[np.ndarray] | None = None,
    ) -> np.ndarray | None:
        self._frame_index += 1
        height, width = frame_bgr.shape[:2]

        if self._frame_index % self.config.background_inference_interval != 0:
            return None

        scale = self.config.background_input_scale
        proc_w = max(96, int(width * scale))
        proc_h = max(96, int(height * scale))
        frame_proc = frame_bgr
        if scale < 0.999:
            frame_proc = self._cv2.resize(
                frame_bgr, (proc_w, proc_h), interpolation=self._cv2.INTER_LINEAR
            )

        frame_rgb = self._cv2.cvtColor(frame_proc, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._segmenter.segment_for_video(mp_image, timestamp_ms)
        if not result.confidence_masks:
            return None

        confidence = result.confidence_masks[0].numpy_view()
        confidence = np.squeeze(confidence).astype(np.float32)
        if confidence.shape != (proc_h, proc_w):
            confidence = self._cv2.resize(
                confidence, (proc_w, proc_h), interpolation=self._cv2.INTER_LINEAR
            )

        confidence = self._cv2.GaussianBlur(confidence, (0, 0), sigmaX=1.3, sigmaY=1.3)
        binary_proc = compute_stable_binary_mask(
            confidence,
            self.config.background_threshold,
            self.config.background_hysteresis,
            self._prev_binary,
        )
        binary_proc = self._refine_binary(binary_proc)

        if self._prev_binary is not None:
            smooth = self.config.background_smoothing
            mix = smooth * binary_proc.astype(np.float32) + (
                1.0 - smooth
            ) * self._prev_binary.astype(np.float32)
            binary_proc = (mix >= 0.5).astype(np.uint8)

        hand_polygons_proc = _rescale_polygons(
            hand_polygons or [], proc_w / float(width), proc_h / float(height)
        )
        binary_proc = force_foreground_by_polygons(
            binary_proc,
            hand_polygons_proc,
            proc_w,
            proc_h,
            self._cv2,
            self.config.hand_mask_scale,
            self.config.hand_mask_dilate,
        )
        self._prev_binary = binary_proc
        binary_full = self._cv2.resize(
            binary_proc.astype(np.float32),
            (width, height),
            interpolation=self._cv2.INTER_LINEAR,
        )
        return self._soften_alpha(binary_full)

    def compose_with_alpha(
        self,
        frame_bgr: np.ndarray,
        alpha: np.ndarray,
        hand_polygons: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        out_alpha = alpha.astype(np.float32, copy=True)
        if out_alpha.shape != (height, width):
            out_alpha = self._cv2.resize(out_alpha, (width, height), interpolation=self._cv2.INTER_LINEAR)

        if hand_polygons:
            hand_binary = (out_alpha >= 0.5).astype(np.uint8)
            hand_binary = force_foreground_by_polygons(
                hand_binary,
                hand_polygons,
                width,
                height,
                self._cv2,
                self.config.hand_mask_scale,
                self.config.hand_mask_dilate,
            )
            out_alpha = self._soften_alpha(hand_binary.astype(np.float32))

        return composite_blur_background(
            frame_bgr, out_alpha, self._cv2, self.config.background_blur_kernel
        )

    def apply(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
        hand_polygons: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        alpha = self.estimate_alpha(frame_bgr, timestamp_ms, hand_polygons)
        if alpha is None:
            return frame_bgr
        return self.compose_with_alpha(frame_bgr, alpha, hand_polygons)

    def close(self) -> None:
        self._segmenter.close()

    def _refine_binary(self, binary: np.ndarray) -> np.ndarray:
        mask = (binary * 255).astype(np.uint8)
        close_kernel = np.ones((3, 3), dtype=np.uint8)
        open_kernel = np.ones((3, 3), dtype=np.uint8)
        mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = self._cv2.morphologyEx(mask, self._cv2.MORPH_OPEN, open_kernel, iterations=1)
        return (mask > 127).astype(np.uint8)

    def _soften_alpha(self, alpha: np.ndarray) -> np.ndarray:
        alpha = alpha.astype(np.float32)
        alpha = self._cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.7, sigmaY=0.7)
        alpha[alpha < 0.08] = 0.0
        alpha[alpha > 0.92] = 1.0
        return alpha


def _scale_polygon(
    polygon: np.ndarray, scale: float, width: int, height: int
) -> np.ndarray:
    center = np.mean(polygon.astype(np.float32), axis=0, keepdims=True)
    expanded = center + (polygon.astype(np.float32) - center) * float(scale)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return np.round(expanded).astype(np.int32)


def _rescale_polygons(
    polygons: list[np.ndarray], sx: float, sy: float
) -> list[np.ndarray]:
    if not polygons:
        return []
    scaled: list[np.ndarray] = []
    for polygon in polygons:
        if polygon is None:
            continue
        points = polygon.astype(np.float32, copy=True)
        points[:, 0] *= sx
        points[:, 1] *= sy
        scaled.append(points)
    return scaled
