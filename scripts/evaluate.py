#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""Evaluate a trained checkpoint on the test split of a dataset (Section 2.7).

In-domain:      --data is the dataset used for training.
Cross-dataset:  --data is a different target dataset (no fine-tuning).

Settings follow the paper: single GPU (device 0), batch 16, FP32, imgsz 640, rect=True,
conf 0.001, NMS IoU 0.7, max_det 300. Precision, Recall, and F1 are taken at the operating
point that maximizes the mean F1-score across classes (Ultralytics default).

Example:
    python scripts/evaluate.py --weights runs/train/stagefm_dfire_s0/weights/best.pt --data configs/fasdd_cv.yaml
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    a = p.parse_args()

    m = YOLO(a.weights).val(
        data=a.data, split="test", batch=16, imgsz=640, half=False, rect=True,
        conf=0.001, iou=0.7, max_det=300, device=0, workers=8, plots=False,
    )
    b = m.box
    p_, r_ = float(b.mp), float(b.mr)
    f1 = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
    fps = 1000.0 / sum(m.speed.values())
    print(f"mAP50 {b.map50:.4f}  mAP50-95 {b.map:.4f}  P {p_:.4f}  R {r_:.4f}  F1 {f1:.4f}")
    for i, c in enumerate(b.ap_class_index):
        print(f"  {m.names[int(c)]} F1 {float(b.f1[i]):.4f}")
    print(f"FPS {fps:.2f}  (evaluation-level throughput: {m.speed})")


if __name__ == "__main__":
    main()
