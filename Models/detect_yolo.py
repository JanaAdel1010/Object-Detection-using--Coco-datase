import os
import cv2
import json
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

input_dir = "val2017"
output_dir = "outputs/yolo"
os.makedirs(output_dir, exist_ok=True)

predictions_yolo = []

for filename in sorted(os.listdir(input_dir))[:20]:
    image_path = os.path.join(input_dir, filename)
    
    # Run YOLO inference
    results = model(image_path, conf=0.5, iou=0.5)

    boxes = []
    scores = []
    img = cv2.imread(image_path)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save image
    cv2.imwrite(os.path.join(output_dir, filename), img)

    # Save predictions
    predictions_yolo.append({
        "image_id": filename,
        "boxes": boxes,
        "scores": scores
    })

# Save predictions to JSON
with open("yolo_predictions.json", "w") as f:
    json.dump(predictions_yolo, f)

print("YOLOv8 inference complete. Predictions saved to yolo_predictions.json")
