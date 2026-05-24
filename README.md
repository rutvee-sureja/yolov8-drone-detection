# YOLOv8 Drone Detection — Real-Time Edge Inference on NVIDIA Jetson Nano

Deployed a real-time UAV detection system on NVIDIA Jetson Nano achieving **mAP@50 = 0.849** and **10+ FPS** live inference using a TensorRT-optimized YOLOv8s model.

Built and led as team lead (5-person international team) during Industrial Project, THWS Schweinfurt — Summer 2026.

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 0.849 |
| Inference speed | 10+ FPS (Jetson Nano) |
| Dataset size | 4,399 images, 12 classes |
| Hardware | NVIDIA Jetson Nano + Arducam IMX477 |

## What it does

Detects and classifies unmanned aerial vehicles (UAVs/drones) in real time from a camera feed. Designed for edge deployment in security and airspace monitoring contexts.

## Stack

Python · PyTorch · YOLOv8 (Ultralytics) · TensorRT · CUDA · OpenCV · GStreamer · NVIDIA Jetson Nano

## Files

- `build_final.py` — main training and model building script
- `test_video.py` — run inference on a video file
- `test_webcam.py` — run inference on live camera feed
- `final_model_dataset2/` — dataset config and structure
- `runs/` — training runs and results

## Team

5-person international team (3 nationalities) | Supervisor: Matthias Schreier, THWS Schweinfurt
