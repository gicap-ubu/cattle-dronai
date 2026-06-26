import os
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

GT_PATH = "~/ganado/labels"
PRED_PATH = "runs_ganado/inferencia_1632/labels"

IOU_THRESHOLDS = [0.3, 0.5, 0.7]
CONF_THRESHOLDS = np.arange(0.05, 0.95, 0.05)

OUT_CSV = "experiments/results_full.csv"
OUT_TXT = "experiments/results_summary.txt"


# =========================
# FUNCIONES AUXILIARES
# =========================

def parse_labels(path):
    """Lee labels YOLO"""
    data = {}
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file)) as f:
                data[file] = [line.strip() for line in f.readlines()]
    return data


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def dummy_match(gt, pred, iou_thr, conf_thr):
    """
    ESTA FUNCIÓN REPRESENTA TU MATCHING ACTUAL
    (usa tu lógica ya implementada)
    """
    tp = max(0, int(len(gt) * 0.6 * (1 - (conf_thr - 0.1))))
    fp = len(pred) - tp if len(pred) > tp else 0
    fn = len(gt) - tp if len(gt) > tp else 0
    return tp, fp, fn


def compute_aupr(precisions, recalls):
    """AUPR simple por trapecios"""
    return np.trapz(precisions, recalls)


# =========================
# MAIN EXPERIMENT
# =========================

def run_experiment():
    gt_files = parse_labels(os.path.expanduser(GT_PATH))
    pred_files = parse_labels(PRED_PATH)

    results = []

    summary_txt = []

    for dist in ["15m", "25m"]:
        summary_txt.append(f"\n========================\nRESULTADOS {dist}\n========================\n")

        for iou in IOU_THRESHOLDS:
            precisions_curve = []
            recalls_curve = []

            for conf in CONF_THRESHOLDS:

                TP = FP = FN = 0

                for file in gt_files.keys():

                    gt = gt_files[file]
                    pred = pred_files.get(file, [])

                    tp, fp, fn = dummy_match(gt, pred, iou, conf)

                    TP += tp
                    FP += fp
                    FN += fn

                P, R, F1 = compute_metrics(TP, FP, FN)

                precisions_curve.append(P)
                recalls_curve.append(R)

                results.append([
                    dist, iou, conf, TP, FP, FN, P, R, F1
                ])

                summary_txt.append(
                    f"CONF={conf:.2f} IOU={iou} "
                    f"P={P:.3f} R={R:.3f} F1={F1:.3f}"
                )

            aupr = compute_aupr(precisions_curve, recalls_curve)

            summary_txt.append(f"\nAUPR (IoU={iou}): {aupr:.4f}\n")

    # =========================
    # GUARDAR CSV
    # =========================

    df = pd.DataFrame(results, columns=[
        "distance", "iou", "conf",
        "TP", "FP", "FN",
        "precision", "recall", "f1"
    ])

    os.makedirs("experiments", exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # =========================
    # GUARDAR TXT
    # =========================

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(summary_txt))

    # =========================
    # PRINT FINAL
    # =========================

    print("\nRESULTADOS GUARDADOS:")
    print("CSV ->", OUT_CSV)
    print("TXT ->", OUT_TXT)


if __name__ == "__main__":
    run_experiment()
