#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""Train YOLOv8n / FEPF / MDDM / Stage-FM from scratch (Section 2.7 and Table 5).

Example:
    python scripts/train.py --model models/yolov8n-stagefm.yaml --data configs/dfire.yaml --seed 0
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model YAML (e.g. yolov8n.yaml, models/yolov8n-stagefm.yaml)")
    p.add_argument("--data", required=True, help="dataset YAML in configs/")
    p.add_argument("--seed", type=int, default=0, help="the paper uses seeds 0, 1, and 2")
    p.add_argument("--device", default="0,1,2,3")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default=None)
    a = p.parse_args()

    YOLO(a.model).train(
        data=a.data, imgsz=640, epochs=300, batch=128, optimizer="AdamW", lr0=0.001,
        patience=50, workers=8, pretrained=False, seed=a.seed, deterministic=True,
        device=a.device, project=a.project, name=a.name,
    )


if __name__ == "__main__":
    main()
