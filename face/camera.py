from __future__ import annotations

from dataclasses import dataclass

from .config import AppConfig


class CameraError(RuntimeError):
    """Raised when webcam initialization or read fails."""


@dataclass
class CameraStream:
    config: AppConfig

    def __post_init__(self) -> None:
        self._cv2 = None
        self._capture = None

    def open(self) -> None:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise CameraError(
                "OpenCV is not installed. Install dependencies from requirements.txt."
            ) from exc

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(self.config.camera_index)
        if not self._capture or not self._capture.isOpened():
            raise CameraError(
                "Webcam is not available.\n"
                "Check camera connection and permissions, then retry with --camera INDEX.\n"
                "On Linux you should have at least one /dev/video* device."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

    def read(self):
        if not self._capture:
            raise CameraError("Camera is not opened.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from webcam.")
        return frame

    def close(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

