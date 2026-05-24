import sys
from ultralytics import YOLO

# Pass model path as argument or use default
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "/Users/rutveesureja/Documents/DroneProject/runs/final_model10/weights/best.pt"

print(f"Using model: {model_path}")

model = YOLO(model_path)
model.predict(
    source=0,
    show=True,
    conf=0.35,
    iou=0.5,
    device="cpu"
)
