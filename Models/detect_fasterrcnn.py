import os
import cv2
import torch
import torchvision
from torchvision import transforms as T
from PIL import Image

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define transform
transform = T.Compose([T.ToTensor()])

# Directory setup
input_dir = "val2017"
output_dir = "Outputs/faster_rcnn"
os.makedirs(output_dir, exist_ok=True)

# Loop through sample images
for filename in os.listdir(input_dir)[:20]:  # run only on 20 for speed
    image_path = os.path.join(input_dir, filename)
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    # Draw boxes
    img_cv = cv2.imread(image_path)
    for i, box in enumerate(prediction["boxes"]):
        score = prediction["scores"][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img_cv)
