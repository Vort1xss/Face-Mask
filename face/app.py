from __future__ import annotations

import time

from .async_runtime import AsyncBackgroundRunner
from .async_runtime import AsyncHandRunner
from .background import BackgroundRemover
from .camera import CameraError, CameraStream
from .config import AppConfig
from .hands import HandTracker
from .mask import MaskRenderer
from .predictor import FacePredictor
from .tracker import FaceMeshTracker
from .virtual_cam import VirtualCamOutput


class FaceApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._cv2 = self._import_cv2()
        self.camera = CameraStream(config)
        self.tracker = FaceMeshTracker(config.model_path)
        self.predictor = FacePredictor(config)
        self.renderer = MaskRenderer(config)
        self.background = (
            BackgroundRemover(config) if config.enable_background_removal else None
        )
        self.hands = HandTracker(config) if config.enable_hand_tracking else None
        self.hand_async = AsyncHandRunner(self.hands) if self.hands is not None else None
        self.background_async = (
            AsyncBackgroundRunner(self.background) if self.background is not None else None
        )
        self.virtual_cam = VirtualCamOutput(config) if config.enable_virtual_cam else None

    @staticmethod
    def _import_cv2():
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Install dependencies from requirements.txt."
            ) from exc
        return cv2

    def run(self) -> int:
        frame_index = 0
        fps = 0.0
        last_fps_ts = time.monotonic()

        try:
            self.camera.open()
            while True:
                frame = self.camera.read()
                if self.config.flip_horizontal:
                    frame = self._cv2.flip(frame, 1)

                now_ms = int(time.monotonic() * 1000)
                gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
                detection = None
                if self.predictor.needs_redetect(frame_index):
                    detection = self.tracker.detect(frame, now_ms).points

                track = self.predictor.update(gray, detection, now_ms)
                hand_polygons = []
                if self.hand_async is not None:
                    self.hand_async.submit(frame, now_ms)
                    hand_polygons = self.hand_async.latest()
                elif self.hands is not None:
                    hand_polygons = self.hands.detect(frame, now_ms)
                render_frame = frame
                if self.background_async is not None:
                    self.background_async.submit(frame, now_ms, hand_polygons)
                    render_frame = self.background_async.render(frame, hand_polygons)
                elif self.background is not None:
                    render_frame = self.background.apply(frame, now_ms, hand_polygons)
                output = self.renderer.draw(render_frame, track)

                if self.config.show_fps:
                    current_ts = time.monotonic()
                    dt = current_ts - last_fps_ts
                    if dt > 0:
                        fps = 1.0 / dt
                    last_fps_ts = current_ts
                    self._cv2.putText(
                        output,
                        f"FPS: {fps:.1f}  source: {track.source}",
                        (10, 24),
                        self._cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        self._cv2.LINE_AA,
                    )

                if self.virtual_cam is not None:
                    self.virtual_cam.send(output)

                if self.config.show_preview:
                    self._cv2.imshow("Face", output)
                    key = self._cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                frame_index += 1
        except KeyboardInterrupt:
            pass
        except CameraError:
            raise
        finally:
            self.camera.close()
            self.tracker.close()
            if self.virtual_cam is not None:
                self.virtual_cam.close()
            if self.background_async is not None:
                self.background_async.close()
            elif self.background is not None:
                self.background.close()
            if self.hand_async is not None:
                self.hand_async.close()
            elif self.hands is not None:
                self.hands.close()
            if self.config.show_preview:
                self._cv2.destroyAllWindows()
        return 0
