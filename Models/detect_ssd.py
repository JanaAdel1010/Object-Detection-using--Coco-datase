import os
import cv2
import torch
import torchvision
from torchvision import transforms
from PIL import Image

# Load model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_dir = "val2017"
output_dir = "outputs/ssd"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir)[:20]:  # Change to [:N] if needed
    path = os.path.join(input_dir, filename)
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    img_cv = cv2.imread(path)
    for i, box in enumerate(predictions["boxes"]):
        score = predictions["scores"][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, filename), img_cv)
