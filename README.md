# YOLOv8 Drone Detection — Custom Model Training & Jetson Nano Camera Pipeline

A 12-class UAV detection system built for the Industrial Project module at THWS Schweinfurt (Summer 2026). The project covers two connected pieces of work: training a custom YOLOv8s detector to **mAP@50 = 0.849**, and bringing up the complete camera-to-inference pipeline on an **NVIDIA Jetson Nano Super** with a CSI-connected Arducam IMX477, including TensorRT FP16 engine conversion.

Built and led as team lead of a 5-person international team (3 nationalities).

---

## Results

### Custom model (trained on Apple M1 MacBook, 8 GB RAM)

| Metric | Value |
|---|---|
| mAP@50 | **0.849** |
| mAP@50–95 | 0.563 |
| Precision | 0.816 |
| Recall | 0.688 |
| Dataset | 4,399 images · 12 classes (3,519 train / 439 val / 441 test) |
| Base model | YOLOv8s (COCO-pretrained), `freeze=10`, 50 epochs, 416 px |

Three model iterations were trained and evaluated:

| Version | Strategy | mAP@50 |
|---|---|---|
| v2 | 81-class (COCO 80 + drone), heavy class imbalance | 0.515 |
| v3 | Reduced class set, remapped IDs | 0.634 |
| **final** | **12-class curated set, `freeze=10` to prevent catastrophic forgetting** | **0.849** |

The final model was validated on live webcam input and recorded drone footage, correctly distinguishing **drone / bird / airplane** — the three classes most easily confused in UAV detection.

### Jetson Nano pipeline

| Item | Detail |
|---|---|
| Platform | NVIDIA Jetson Nano Super · Ubuntu 22.04 · JetPack (CUDA, cuDNN, TensorRT) |
| Camera | Arducam 12 MP IMX477 via CSI · RG10 10-bit raw Bayer |
| Pipeline | `nvarguscamerasrc` → NV12 (1920×1080) → `nvvidconv` → BGRx (640×480) → `videoconvert` → BGR → `appsink` |
| Inference | YOLOv8s validated on Jetson as both PyTorch `.pt` and TensorRT `.engine` (FP16, layer fusion) |

End-to-end pipeline validation on the Jetson was performed with the COCO-pretrained YOLOv8s model; the design allows switching to the custom-trained weights via a single command-line argument. Real-time inference of the custom model on the MacBook CPU ran at 3–5 FPS.

---

## Engineering problems solved on the Jetson

| Challenge | Root cause | Solution |
|---|---|---|
| Solid green camera output | OpenCV cannot decode RG10 raw Bayer | Replaced `v4l2src` with `nvarguscamerasrc`, which handles ISP processing and Bayer demosaicing via the Argus API |
| GStreamer negotiation failure | `v4l2src` could not negotiate RG10 with `bayer2rgb` | Rebuilt pipeline around `nvarguscamerasrc` |
| System overcurrent warning | 4K capture + YOLO inference exceeded the power budget | Capped capture at 1080p, inference input at 640 px, added frame skipping to stay within thermal limits |
| Green frames on startup | Camera buffer not flushed at init | 30-frame buffer flush on initialisation |
| Window would not close | OpenCV ignores OS close events | Keyboard quit handler + `pkill` fallback |
| Focus blur | IMX477 has a manual focus ring | Physical lens adjustment |

---

## Stack

Python · PyTorch · Ultralytics YOLOv8 · OpenCV · GStreamer · TensorRT (FP16) · CUDA · NVIDIA Jetson Nano Super · Arducam IMX477 (CSI)

---

## Files

| File | Purpose |
|---|---|
| `build_final.py` | Dataset construction (custom drone/bird/airplane + curated COCO subset), 80/10/10 split, `data.yaml` generation, and YOLOv8s training |
| `coco_info.py` | Inspects COCO annotation structure and category IDs |
| `test_video.py` | Runs inference on a recorded video file |
| `test_webcam.py` | Runs inference on a live camera feed |
| `final_model_dataset2/` | Dataset configuration and structure |
| `runs/` | Training runs, weights, and result plots |

> **Note on scope:** the Jetson-side camera pipeline and TensorRT conversion were developed and validated directly on the Jetson device during the project and are documented in the project report; that code is not included in this repository.

---

## My contribution

Team lead, and personally responsible for:

- Dataset design and construction across three iterations — class selection, ID remapping, COCO subset curation, train/val/test splitting
- Training and evaluation of all three model versions, including the diagnosis that class imbalance capped v2 and that `freeze=10` was needed to preserve COCO backbone features
- Jetson Nano camera bring-up: Arducam IMX477 CSI setup, `v4l2-ctl` verification, GStreamer pipeline debugging and tuning
- TensorRT FP16 engine conversion and end-to-end pipeline validation on the Jetson
- Project agreement, milestone planning, and team coordination across a 5-person team from 3 nationalities

---

## Project context

Industrial Project · B.Eng. Mechatronics (International Programme) · Technische Hochschule Würzburg-Schweinfurt
Supervisor: Matthias Schreier · Summer semester 2026 · Grade: 1.3
