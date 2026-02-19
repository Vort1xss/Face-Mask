from __future__ import annotations

import argparse
import sys

from face.app import FaceApp
from face.camera import CameraError
from face.config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime head/face tracking with a pixelated privacy blur."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument(
        "--mask-color",
        type=str,
        default="0,255,0",
        help="Deprecated compatibility flag (no-op). Previous mask BGR color (example: 0,255,0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Global face blur alpha in range 0..1.",
    )
    parser.add_argument(
        "--mask-scale",
        type=float,
        default=1.18,
        help="Scale of the face mask polygon (1.0..1.8).",
    )
    parser.add_argument(
        "--face-pixel-size",
        type=int,
        default=16,
        help="Pixel block size for face blur (4..24).",
    )
    parser.add_argument(
        "--loss-ttl-ms",
        type=int,
        default=200,
        help="How long to keep predicted mask after tracking loss.",
    )
    parser.add_argument(
        "--fade-out-ms",
        type=int,
        default=250,
        help="Fade-out duration after loss TTL is exceeded.",
    )
    parser.add_argument(
        "--re-detect-interval",
        type=int,
        default=2,
        help="Run full FaceMesh detection every N frames.",
    )
    parser.add_argument(
        "--flow-min-ratio",
        type=float,
        default=0.55,
        help="Minimum valid optical-flow point ratio to accept flow update.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task model.",
    )
    parser.add_argument(
        "--bg-model-path",
        type=str,
        default="models/selfie_segmenter.tflite",
        help="Path to MediaPipe selfie segmenter model.",
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        default="255,255,0",
        help="Legacy background color option (kept for compatibility).",
    )
    parser.add_argument(
        "--bg-blur-kernel",
        type=int,
        default=61,
        help="Strong background blur kernel size (odd number, e.g. 61).",
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.2,
        help="Foreground confidence threshold for person mask (0..1).",
    )
    parser.add_argument(
        "--bg-hysteresis",
        type=float,
        default=0.08,
        help="Stability hysteresis around background threshold (0..0.35).",
    )
    parser.add_argument(
        "--bg-interval",
        type=int,
        default=2,
        help="Run background segmentation every N frames.",
    )
    parser.add_argument(
        "--bg-scale",
        type=float,
        default=0.6,
        help="Background segmentation input scale (0.3..1.0).",
    )
    parser.add_argument(
        "--bg-smoothing",
        type=float,
        default=0.65,
        help="Temporal smoothing for segmentation mask (0..1).",
    )
    parser.add_argument(
        "--hand-model-path",
        type=str,
        default="models/hand_landmarker.task",
        help="Path to MediaPipe hand_landmarker.task model.",
    )
    parser.add_argument(
        "--hand-mask-scale",
        type=float,
        default=1.35,
        help="Scale of hand foreground keep-mask (1.0..2.5).",
    )
    parser.add_argument(
        "--hand-mask-dilate",
        type=int,
        default=2,
        help="Hand keep-mask dilation iterations.",
    )
    parser.add_argument(
        "--hand-interval",
        type=int,
        default=2,
        help="Run hand detection every N frames.",
    )
    parser.add_argument(
        "--hand-scale",
        type=float,
        default=0.65,
        help="Hand detection input scale (0.35..1.0).",
    )
    parser.add_argument(
        "--disable-bg-remove",
        action="store_true",
        help="Disable person/background segmentation and replacement.",
    )
    parser.add_argument(
        "--disable-hands",
        action="store_true",
        help="Disable hand tracking assistance for background separation.",
    )
    parser.add_argument(
        "--virtual-cam",
        action="store_true",
        help="Output processed frames to virtual webcam device for OBS.",
    )
    parser.add_argument(
        "--virtual-cam-device",
        type=str,
        default="/dev/video10",
        help="Virtual webcam device path (Linux v4l2loopback), e.g. /dev/video10.",
    )
    parser.add_argument(
        "--virtual-cam-backend",
        type=str,
        default="v4l2loopback",
        help="pyvirtualcam backend name (default: v4l2loopback).",
    )
    parser.add_argument(
        "--virtual-cam-fps",
        type=float,
        default=30.0,
        help="Virtual webcam output FPS.",
    )
    parser.add_argument("--show-fps", action="store_true", help="Show FPS overlay.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable local OpenCV preview window (useful for OBS-only mode).",
    )
    parser.add_argument(
        "--no-flip",
        action="store_true",
        help="Disable horizontal mirror flip.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = AppConfig.from_args(args)
        app = FaceApp(config)
        return app.run()
    except (CameraError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
