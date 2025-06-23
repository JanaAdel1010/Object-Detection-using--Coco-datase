import os
import cv2
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import json


# Load model 
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Image transform
transform = T.Compose([T.ToTensor()])

# Directory setup
input_dir = "val2017"
output_dir = "outputs/faster_rcnn"
os.makedirs(output_dir, exist_ok=True)

# Inference loop and save
predictions_rcnn = []

for filename in sorted(os.listdir(input_dir))[:20]:  # First 20 images only
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = []
    scores = []
    img_cv = cv2.imread(image_path)

    for i, box in enumerate(prediction["boxes"]):
        score = prediction["scores"][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save image with boxes
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img_cv)

    # Collect predictions
    predictions_rcnn.append({
        "image_id": filename,
        "boxes": boxes,
        "scores": scores
    })


# Save predictions to JSON
with open("rcnn_predictions.json", "w") as f:
    json.dump(predictions_rcnn, f)

print("Faster R-CNN inference complete. Predictions saved to rcnn_predictions.json")
