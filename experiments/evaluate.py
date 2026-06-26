import os
import glob

IOU_THRESHOLD = 0.3

def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return (min(xs), min(ys), max(xs), max(ys))

def load_gt(file_path):
    """
    Convierte cada polígono del ground truth a una bounding box.
    """
    boxes = []

    with open(file_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))

            if len(parts) < 5:
                continue

            coords = parts[1:]  # quitar clase

            boxes.append(polygon_to_bbox(coords))

    return boxes


def load_pred(file_path):
    """
    Lee predicciones YOLO:
    clase x_centro y_centro ancho alto confianza
    """
    boxes = []

    if not os.path.exists(file_path):
        return boxes

    with open(file_path, "r") as f:
        for line in f:

            parts = list(map(float, line.strip().split()))

            if len(parts) < 6:
                continue

            x = parts[1]
            y = parts[2]
            w = parts[3]
            h = parts[4]

            xmin = x - w / 2
            ymin = y - h / 2
            xmax = x + w / 2
            ymax = y + h / 2

            boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


def evaluate(gt_dir, pred_dir, prefix=None):

    TP = 0
    FP = 0
    FN = 0

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.txt")))

    if prefix is not None:
        gt_files = [
            f for f in gt_files
            if os.path.basename(f).startswith(prefix)
        ]

    print(f"\nEvaluando {len(gt_files)} imágenes ({prefix if prefix else 'TOTAL'})")

    for gt_file in gt_files:

        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)

        gt_boxes = load_gt(gt_file)
        pred_boxes = load_pred(pred_file)

        matched_preds = set()

        for gt_box in gt_boxes:

            best_iou = 0
            best_pred_idx = -1

            for pred_idx, pred_box in enumerate(pred_boxes):

                if pred_idx in matched_preds:
                    continue

                current_iou = iou(gt_box, pred_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_idx = pred_idx

            if best_iou >= IOU_THRESHOLD:
                TP += 1
                matched_preds.add(best_pred_idx)
            else:
                FN += 1

        FP += len(pred_boxes) - len(matched_preds)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return TP, FP, FN, precision, recall, f1


GT_DIR = "/home/beatriz/ganado/labels"
PRED_DIR = "/home/beatriz/yolov5/runs_ganado/inferencia_1632/labels"

print("\n========================")
print("RESULTADOS TOTALES")
print("========================")

tp, fp, fn, p, r, f1 = evaluate(GT_DIR, PRED_DIR)

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1-score: {f1:.4f}")

print("\n========================")
print("RESULTADOS 15m")
print("========================")

tp, fp, fn, p, r, f1 = evaluate(GT_DIR, PRED_DIR, prefix="15m")

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1-score: {f1:.4f}")

print("\n========================")
print("RESULTADOS 25m")
print("========================")

tp, fp, fn, p, r, f1 = evaluate(GT_DIR, PRED_DIR, prefix="25m")

print(f"TP: {tp}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1-score: {f1:.4f}")
