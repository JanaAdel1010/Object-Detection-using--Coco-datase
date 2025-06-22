import os
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv5s model
model = YOLO('yolov5s.pt')  # can use yolov5m.pt or yolov8s.pt

input_dir = "val2017"
output_dir = "outputs/yolo"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir)[:20]:
    path = os.path.join(input_dir, filename)

    results = model(path)
    results[0].save(filename=os.path.join(output_dir, filename))
