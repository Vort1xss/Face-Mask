Face

Face is a real-time streaming tool that hides your face and outputs the processed video as a virtual camera for OBS.

It captures webcam video, detects facial landmarks, applies a stabilized mask, optionally performs background removal or blur, and streams the final result to OBS via a virtual camera.

Features

Real-time face detection and tracking

Stable face mask overlay

Optical flow smoothing between detections

Smooth fade-out when face is lost

Optional background segmentation (blur/remove)

Hand-aware processing

Virtual camera output (e.g. /dev/video10)

How It Works

Captures frames from your webcam.

Detects face and facial landmarks.

Uses tracking between full detections for smooth motion.

Predicts and fades mask if face temporarily disappears.

Optionally separates subject from background.

Sends processed frames to a virtual camera device.

Requirements

Linux (v4l2loopback for virtual camera)

Webcam

OBS Studio

Python 3.x

OpenCV

MediaPipe (or other landmark detection backend)

Installation
git clone https://github.com/yourusername/face.git
cd face
pip install -r requirements.txt


Load virtual camera module:

sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FaceCam" exclusive_caps=1

Usage
python main.py


In OBS:

Add a Video Capture Device

Select FaceCam (or /dev/video10)

Use Cases

Anonymous streaming

Live camera replacement instead of static overlay

Privacy-focused content creation

License

MIT License
