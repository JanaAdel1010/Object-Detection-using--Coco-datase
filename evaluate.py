import json
import os
from pycocotools.coco import COCO


# 1. IoU Function
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# 2. Evaluation Function
def evaluate(predictions, ground_truths, iou_threshold=0.5):
    total_tp = 0
    total_pred = 0
    total_gt = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred["boxes"]
        gt_boxes = gt["boxes"]
        matched = set()
        tp = 0

        for pbox in pred_boxes:
            for i, gtbox in enumerate(gt_boxes):
                iou = compute_iou(pbox, gtbox)
                if iou >= iou_threshold and i not in matched:
                    matched.add(i)
                    tp += 1
                    break

        total_tp += tp
        total_pred += len(pred_boxes)
        total_gt += len(gt_boxes)

    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 3. Load Ground Truths from COCO Annotations
def load_ground_truths(ann_file, image_filenames):
    coco = COCO(ann_file)
    ground_truths = []

    for fname in image_filenames:
        image_id = int(os.path.splitext(fname)[0])
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])

        ground_truths.append({
            "image_id": fname,
            "boxes": boxes
        })
    return ground_truths


def main():
     # Paths
    image_dir = "val2017"
    ann_file = "annotations/instances_val2017.json"
    rcnn_file = "rcnn_predictions.json"
    ssd_file = "ssd_predictions.json"
    yolo_file = "yolo_predictions.json"

    # Load predictions
    with open(rcnn_file, "r") as f:
        predictions_rcnn = json.load(f)
    with open(ssd_file, "r") as f:
        predictions_ssd = json.load(f)
    with open(yolo_file, "r") as f:
        predictions_yolo = json.load(f)

    # Get first 20 image filenames
    image_filenames = sorted(os.listdir(image_dir))[:20]

    # Load ground truth
    ground_truths = load_ground_truths(ann_file, image_filenames)

    # Evaluate all models
    results_rcnn = evaluate(predictions_rcnn, ground_truths)
    results_ssd = evaluate(predictions_ssd, ground_truths)
    results_yolo = evaluate(predictions_yolo, ground_truths)

    # Print results
    print("Faster R-CNN Results:")
    print("Precision:", results_rcnn["precision"])
    print("Recall:", results_rcnn["recall"])
    print("F1 Score:", results_rcnn["f1"])

    print("\nSSD Results:")
    print("Precision:", results_ssd["precision"])
    print("Recall:", results_ssd["recall"])
    print("F1 Score:", results_ssd["f1"])

    print("\nYOLOv8 Results:")
    print("Precision:", results_yolo["precision"])
    print("Recall:", results_yolo["recall"])
    print("F1 Score:", results_yolo["f1"])


if __name__ == "__main__":
    main()
