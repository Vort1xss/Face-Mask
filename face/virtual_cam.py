from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AppConfig


class VirtualCamOutput:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pyvirtualcam = None
        self._camera = None

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._camera is None:
            self._open(frame_bgr)
        self._camera.send(frame_bgr)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    def _open(self, frame_bgr: np.ndarray) -> None:
        try:
            import pyvirtualcam  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "pyvirtualcam is not installed. Reinstall dependencies from requirements.txt."
            ) from exc

        height, width = frame_bgr.shape[:2]
        self._pyvirtualcam = pyvirtualcam
        backend = self._normalize_optional(self._config.virtual_cam_backend)
        device = self._normalize_optional(self._config.virtual_cam_device)
        if backend == "v4l2loopback" and device is not None:
            if not Path(device).exists():
                raise RuntimeError(
                    f"Virtual camera device {device} does not exist.\n"
                    "Create it first (Linux):\n"
                    "sudo modprobe v4l2loopback devices=1 video_nr=10 "
                    "card_label='Face Virtual Cam' exclusive_caps=1"
                )
        self._camera = pyvirtualcam.Camera(
            width=width,
            height=height,
            fps=float(self._config.virtual_cam_fps),
            fmt=pyvirtualcam.PixelFormat.BGR,
            device=device,
            backend=backend,
            print_fps=False,
        )

    @staticmethod
    def _normalize_optional(value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        return value or None
