import json, os
from pycocotools.coco import COCO

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1, area2 = (x2 - x1) * (y2 - y1), (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate(predictions, ground_truths, iou_threshold=0.5):
    tp, total_pred, total_gt = 0, 0, 0
    for pred, gt in zip(predictions, ground_truths):
        matched = set()
        for pbox in pred["boxes"]:
            for i, gtbox in enumerate(gt["boxes"]):
                if compute_iou(pbox, gtbox) >= iou_threshold and i not in matched:
                    matched.add(i)
                    tp += 1
                    break
        total_pred += len(pred["boxes"])
        total_gt += len(gt["boxes"])
    precision = tp / total_pred if total_pred else 0
    recall = tp / total_gt if total_gt else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1}

def load_ground_truths(dataset, image_filenames):
    if dataset == "coco":
        coco = COCO("annotations/instances_val2017.json")
        gt = []
        for fname in image_filenames:
            image_id = int(os.path.splitext(fname)[0])
            anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
            boxes = [[x, y, x+w, y+h] for a in anns for x, y, w, h in [a["bbox"]]]
            gt.append({"image_id": fname, "boxes": boxes})
        return gt
    elif dataset == "voc":
        import xml.etree.ElementTree as ET
        gt = []
        for fname in image_filenames:
            xml_path = f"VOCdevkit/VOC2007/Annotations/{fname.replace('.jpg', '.xml')}"
            tree = ET.parse(xml_path)
            boxes = []
            for obj in tree.findall("object"):
                b = obj.find("bndbox")
                boxes.append([int(b.find("xmin").text), int(b.find("ymin").text),
                              int(b.find("xmax").text), int(b.find("ymax").text)])
            gt.append({"image_id": fname, "boxes": boxes})
        return gt

def main():
    dataset = "voc"  # change to "coco" as needed
    image_dir = "val2017" if dataset == "coco" else "VOCdevkit/VOC2007/JPEGImages"
    image_filenames = sorted(os.listdir(image_dir))[:20]
    ground_truths = load_ground_truths(dataset, image_filenames)

    models = ["rcnn", "ssd", "yolo"]
    for m in models:
        with open(f"{m}_predictions_{dataset}.json") as f:
            predictions = json.load(f)
        r = evaluate(predictions, ground_truths)
        print(f"\n{m.upper()} on {dataset.upper()}")
        print(f"Precision: {r['precision']:.3f}, Recall: {r['recall']:.3f}, F1: {r['f1']:.3f}")

if __name__ == "__main__":
    main()
