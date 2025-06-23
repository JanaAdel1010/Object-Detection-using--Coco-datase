import os
import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import SSD300_VGG16_Weights
from PIL import Image
import json


# Load SSD model (no warnings)
weights = SSD300_VGG16_Weights.DEFAULT
model = torchvision.models.detection.ssd300_vgg16(weights=weights)
model.eval()

# Preprocessing transform
transform = weights.transforms()

# Directory setup
input_dir = "val2017"
output_dir = "outputs/ssd"
os.makedirs(output_dir, exist_ok=True)

# Inference and save predictions
predictions_ssd = []

for filename in sorted(os.listdir(input_dir))[:20]:  # First 20 images
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
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save image with boxes
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img_cv)

    # Collect predictions
    predictions_ssd.append({
        "image_id": filename,
        "boxes": boxes,
        "scores": scores
    })

# Save to JSON
with open("ssd_predictions.json", "w") as f:
    json.dump(predictions_ssd, f)

print("SSD inference complete. Predictions saved to ssd_predictions.json")
