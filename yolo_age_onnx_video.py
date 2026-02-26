import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------
VIDEO_IN = r"D:\Projects\Age group\yolo_age_project\input.mp4"
VIDEO_OUT = "output_age.mp4"
AGE_LABELS = ["Teen", "Adult", "Senior"]

# -----------------------
# Load YOLO
# -----------------------
yolo = YOLO("yolov8n.pt")

# -----------------------
# Load ONNX Age Model
# -----------------------
session = ort.InferenceSession("models/age_model.onnx")
input_name = session.get_inputs()[0].name

# -----------------------
# Preprocess function
# -----------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# -----------------------
# Video I/O
# -----------------------
cap = cv2.VideoCapture(VIDEO_IN)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open video file: {VIDEO_IN}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 25

print("Frame size:", w, h, "FPS:", fps)

OUTPUT_VIDEO = r"D:\Projects\Age group\yolo_age_project\output_age.avi"

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (w, h)
)

print("VideoWriter opened:", out.isOpened())

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO + age prediction here ---

    out.write(frame)
    frame_count += 1

cap.release()
out.release()

print("Frames written:", frame_count)
print("✅ Video saved at:", OUTPUT_VIDEO)
