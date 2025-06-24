import os, cv2, json
from ultralytics import YOLO

dataset = "voc"
input_dir = "val2017" if dataset == "coco" else "VOCdevkit/VOC2007/JPEGImages"
output_dir = f"outputs/{dataset}/yolo"
json_output = f"yolo_predictions_{dataset}.json"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")
predictions = []

for filename in sorted(os.listdir(input_dir))[:20]:
    path = os.path.join(input_dir, filename)
    results = model(path, conf=0.5, iou=0.5)
    img = cv2.imread(path)
    boxes, scores = [], []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(os.path.join(output_dir, filename), img)
    predictions.append({"image_id": filename, "boxes": boxes, "scores": scores})

with open(json_output, "w") as f:
    json.dump(predictions, f)
print(f"YOLOv8 finished on {dataset.upper()} → {json_output}")
