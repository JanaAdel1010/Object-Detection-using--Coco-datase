import os, cv2, json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

dataset = "voc"
input_dir = "val2017" if dataset == "coco" else "VOCdevkit/VOC2007/JPEGImages"
output_dir = f"outputs/{dataset}/ssd"
json_output = f"ssd_predictions_{dataset}.json"
os.makedirs(output_dir, exist_ok=True)

weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
model.eval()
transform = weights.transforms()

predictions = []
for filename in sorted(os.listdir(input_dir))[:50]:
    path = os.path.join(input_dir, filename)
    image = Image.open(path).convert("RGB")
    tensor = transform(image)

    with torch.no_grad():
        result = model([tensor])[0]

    boxes, scores = [], []
    img = cv2.imread(path)
    for i, box in enumerate(result["boxes"]):
        score = result["scores"][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, filename), img)
    predictions.append({"image_id": filename, "boxes": boxes, "scores": scores})

with open(json_output, "w") as f:
    json.dump(predictions, f)
print(f"SSD finished on {dataset.upper()} → {json_output}")
