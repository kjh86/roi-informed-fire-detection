# roi-informed-fire-detection

Official code for the paper:

> **ROI-Based Visual Characteristic Analysis and Feature Enhancement for Fire and Smoke Detection across Multiple Data Distributions**
> Jinho Kim and Gye-Young Kim, *Fire* (MDPI), 2026 (under review).

This repository provides

1. **ROI-based visual characteristic analysis** of fire, smoke, and background regions (Sobel Mean, Edge Density, DWT, FFT, GLCM, Laplacian Variance), including image-level cluster-bootstrap confidence intervals, Hedges' g, and the zero-overlap background robustness check;
2. **Feature enhancement modules** for YOLOv8n — Fixed-Edge Prior Fusion (**FEPF**), Multi-Dilation Depthwise Mixer (**MDDM**), and their stage-wise integration (**Stage-FM**, P3: FEPF / P4: MDDM);
3. **Training and evaluation scripts** for in-domain and cross-dataset (source → target) evaluation.

---

## Repository structure

```
roi-informed-fire-detection/
├── roi_analysis/
│   ├── roi_multifeature_analysis.py   # ROI extraction + Sobel/Canny/DWT/FFT/GLCM/Laplacian features
│   ├── roi_statistics.py              # image-level cluster bootstrap 95% CI, Hedges' g (Table 8)
│   └── zero_overlap_background.py     # zero-overlap background robustness analysis (Tables S2–S3)
├── models/
│   ├── fire_modules.py                # FEPF and MDDM
│   ├── yolov8n-fepf.yaml              # FEPF at P3
│   ├── yolov8n-mddm.yaml              # MDDM at P4
│   └── yolov8n-stagefm.yaml           # Stage-FM (P3: FEPF, P4: MDDM)
├── tools/
│   └── install_modules.py             # registers FEPF/MDDM in the installed Ultralytics package
├── scripts/
│   ├── train.py                       # training (from scratch, seeds 0/1/2)
│   └── evaluate.py                    # in-domain / cross-dataset evaluation, F1 and FPS
└── configs/                           # dataset YAMLs (dfire, fasdd_cv, fasd)
```

## Requirements

| Item | Version used in the paper |
|---|---|
| Python | 3.9.13 |
| PyTorch | 2.0.0 + CUDA 11.7 |
| Ultralytics | 8.4.17 |
| GPU | NVIDIA GeForce RTX 3090 |

```bash
git clone https://github.com/kjh86/roi-informed-fire-detection.git
cd roi-informed-fire-detection
pip install -r requirements.txt
python tools/install_modules.py      # register FEPF and MDDM in Ultralytics (run once)
```

`install_modules.py` copies `models/fire_modules.py` into the installed Ultralytics package and adds FEPF and MDDM to the module registry (`base_modules`) in `ultralytics/nn/tasks.py`, keeping a backup as `tasks.py.bak`. The baseline is the standard `yolov8n.yaml` shipped with Ultralytics.

To verify the installation, the parameter counts (nc = 2) should match Table 4 of the paper:

```bash
python -c "
from ultralytics.nn.tasks import DetectionModel as D
for y in ['yolov8n.yaml','models/yolov8n-fepf.yaml','models/yolov8n-mddm.yaml','models/yolov8n-stagefm.yaml']:
    m = D(y, nc=2, verbose=False); print(y, sum(p.numel() for p in m.parameters()))
"
```

| Model | Parameters |
|---|---|
| YOLOv8n | 3,011,238 |
| FEPF | 3,017,710 |
| MDDM | 3,032,393 |
| Stage-FM | 3,038,865 |

## Datasets

The datasets are **not redistributed** in this repository. Please download them from the original sources:

| Dataset | Reference | Link |
|---|---|---|
| D-Fire | de Venâncio et al., *Neural Comput. Appl.* 2022 | [Official GitHub](https://github.com/gaia-solutions-on-demand/DFireDataset) (the Kaggle version linked there was used: [Kaggle](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)) |
| FASDD_CV | Wang et al., *Geo-spatial Inf. Sci.* 2025 | [GitHub](https://github.com/openrsgis/FASDD) · [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda) (computer-vision subset, FASDD_CV) |
| FASD | Yao et al., *SIViP* 2025 | [Google Drive (provided by the authors)](https://drive.google.com/drive/folders/15GeO6cK_aepnn2xkSpZ487_901ZhYgfJ?usp=sharing) |

Convert each dataset to the YOLO format and use the original train / validation / test splits (Table 2 of the paper). Set the absolute dataset path in `configs/*.yaml` (`path:`); relative paths are resolved by Ultralytics against its `datasets_dir` setting.

```
datasets/
├── D-Fire/   {images,labels}/{train,val,test}
├── FASDD_CV/ {images,labels}/{train,val,test}
└── FASD/     {images,labels}/{train,val,test}
```

## 1. ROI-based visual characteristic analysis

Only the **training split** is used. For each dataset, 3,000 ROIs per class (fire, smoke, background) are collected with seed 42 and converted to 128 × 128 grayscale. Background ROIs in the main analysis are accepted when their maximum IoU with size-filtered annotations is ≤ 0.05.

```bash
# ROI extraction and feature computation (repeat for each dataset)
python roi_analysis/roi_multifeature_analysis.py --dataset-root datasets/D-Fire --splits train \
    --max-samples 3000 --patch-size 128 --seed 42 --output-dir data/multifeature_train3000_235/ds2

# Image-level cluster bootstrap 95% CIs (2,000 resamples, seed 42) and Hedges' g
python roi_analysis/roi_statistics.py --root data

# Zero-overlap background robustness analysis (fire/smoke ROIs unchanged)
ROI_DATA_ROOT=data python roi_analysis/zero_overlap_background.py
```

`roi_statistics.py` and `zero_overlap_background.py` expect the feature folders `ds2` (D-Fire), `ds3` (FASDD_CV), and `ds5` (FASD) under `data/multifeature_train3000_235/`; `zero_overlap_background.py` additionally expects the datasets under `data/2`, `data/3`, and `data/5`.

## 2. Training

All core models are trained from scratch with seeds 0, 1, and 2.

```bash
python scripts/train.py --model yolov8n.yaml                --data configs/dfire.yaml --seed 0   # baseline
python scripts/train.py --model models/yolov8n-fepf.yaml    --data configs/dfire.yaml --seed 0
python scripts/train.py --model models/yolov8n-mddm.yaml    --data configs/dfire.yaml --seed 0
python scripts/train.py --model models/yolov8n-stagefm.yaml --data configs/dfire.yaml --seed 0
```

Training settings: imgsz 640, 300 epochs, batch 128, AdamW, lr0 0.001, early-stopping patience 50, 8 workers.

## 3. Evaluation

```bash
# In-domain (same dataset as training)
python scripts/evaluate.py --weights runs/train/<run>/weights/best.pt --data configs/dfire.yaml

# Cross-dataset (source -> target, no fine-tuning)
python scripts/evaluate.py --weights runs/train/<run>/weights/best.pt --data configs/fasdd_cv.yaml
```

Evaluation settings used for the reported accuracy and FPS: single RTX 3090 (device 0), batch 16, FP32 (`half=False`), `imgsz=640`, `rect=True`, `conf=0.001`, `iou=0.7`, `max_det=300`. Precision, Recall, and F1-score are reported at the operating point that maximizes the mean F1-score across classes. FPS is evaluation-level throughput (1000 / sum of per-image stage times), not pure forward-pass latency.

## Citation

If you find this work useful, please cite:

```bibtex
@article{kim2026roiinformed,
  title   = {ROI-Based Visual Characteristic Analysis and Feature Enhancement for Fire and Smoke Detection across Multiple Data Distributions},
  author  = {Kim, Jinho and Kim, Gye-Young},
  journal = {Fire},
  year    = {2026},
  note    = {Under review}
}
```

## License

This project is released under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0), consistent with [Ultralytics](https://github.com/ultralytics/ultralytics), on which it depends. The datasets are not redistributed and remain subject to the licenses of their original providers.

## Acknowledgments

This implementation is built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
