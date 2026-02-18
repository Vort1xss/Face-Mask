# Face

Realtime head/face tracking app that overlays a solid green privacy mask, removes the background, and uses prediction to prevent mask jumps during short tracking loss.

## Features

- Realtime webcam processing.
- Face oval tracking via MediaPipe FaceLandmarker.
- Green mask overlay on the detected face region.
- Person segmentation and strong blurred background replacement.
- Hand tracking that forces hands to stay in foreground (not replaced by background).
- Anti-slip strategy:
  - Optical flow tracking between detections.
  - Periodic re-detect for drift correction.
  - `freeze+predict` for 200 ms on temporary loss.
  - Smooth fade-out after TTL expires.

## Requirements

- Linux/macOS/Windows with webcam.
- Python 3.11 (recommended for MediaPipe compatibility).

## Setup

```bash
cd ~/Desktop/Project/Face
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download model:

```bash
mkdir -p models
curl -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task -o models/face_landmarker.task
curl -L https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite -o models/selfie_segmenter.tflite
curl -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -o models/hand_landmarker.task
```

## Run

```bash
python main.py --camera 0 --show-fps
```

Useful flags:

- `--mask-color 0,255,0`
- `--mask-scale 1.18`
- `--loss-ttl-ms 200`
- `--fade-out-ms 250`
- `--re-detect-interval 2`
- `--flow-min-ratio 0.55`
- `--model-path models/face_landmarker.task`
- `--bg-model-path models/selfie_segmenter.tflite`
- `--bg-blur-kernel 61`
- `--bg-threshold 0.2`
- `--bg-hysteresis 0.08`
- `--bg-interval 2`
- `--bg-scale 0.6`
- `--bg-smoothing 0.65`
- `--hand-model-path models/hand_landmarker.task`
- `--hand-mask-scale 1.35`
- `--hand-mask-dilate 2`
- `--hand-interval 2`
- `--hand-scale 0.65`
- `--disable-hands`
- `--disable-bg-remove`
- `--virtual-cam`
- `--virtual-cam-device /dev/video10`
- `--virtual-cam-backend v4l2loopback`
- `--virtual-cam-fps 30`
- `--no-preview`
- `--no-flip`

Press `q` or `Esc` to exit.

Performance preset for weak CPUs:

```bash
python main.py --camera 0 --width 640 --height 360 --re-detect-interval 2 --bg-interval 3 --bg-scale 0.5 --hand-interval 3 --hand-scale 0.55
```

## OBS Virtual Webcam (Linux)

1. Install loopback driver (Arch):

```bash
yay -S v4l2loopback-dkms v4l2loopback-utils
```

2. Create virtual webcam device:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Face Virtual Cam" exclusive_caps=1
```

3. Run Face in virtual camera mode:

```bash
python main.py --camera 0 --virtual-cam --virtual-cam-device /dev/video10 --virtual-cam-backend v4l2loopback --no-preview --show-fps
```

4. In OBS:
- Add source `Video Capture Device`.
- Select device `Face Virtual Cam` (or `/dev/video10`).

If OBS is Flatpak and does not see the device, allow device access:

```bash
flatpak override --user --device=all com.obsproject.Studio
```

## Testing

```bash
pytest -q
```

## Troubleshooting

- If webcam fails to open:
  - Verify camera is connected.
  - Check device permission.
  - Try another index: `--camera 1`.
  - On Linux, ensure at least one `/dev/video*` device exists.
- If `mediapipe` install fails:
  - Confirm Python is 3.11 in the active venv.
# Face-Mask
