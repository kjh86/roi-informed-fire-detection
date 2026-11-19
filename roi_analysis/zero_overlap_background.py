#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""Zero-overlap background robustness analysis (Section 3.1, Tables S2 and S3).

Only the background ROIs of datasets 2 (D-Fire), 3 (FASDD_CV), and 5 (FASD) are resampled.
The original fire and smoke rows (training split, seed 42, 3,000 per class) are kept unchanged.
Background patches (48–160 px) have zero pixel intersection with every annotated box,
including boxes smaller than 20 px. Results are written to multifeature_train3000_235_bg0/
and the original folders are not modified.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import os

# Root folder that contains the datasets (2 = D-Fire, 3 = FASDD_CV, 5 = FASD) and the
# original ROI feature folder. Override with the ROI_DATA_ROOT environment variable.
ROOT = Path(os.environ.get("ROI_DATA_ROOT", "./data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from roi_multifeature_analysis import (  # noqa: E402
    band_energies_from_profile,
    collect_image_label_pairs,
    crop_safe,
    fft_log_magnitude,
    glcm_properties,
    gradient_edge_features,
    preprocess_roi,
    radial_profile,
    sample_zero_overlap_patch,
    wavelet_level_energies,
    yolo_all_boxes_xyxy,
    yolo_boxes_to_pixels,
)

OLD = ROOT / "multifeature_train3000_235"
OUT = ROOT / "multifeature_train3000_235_bg0"
DATASETS = [
    ("ds2", "D-Fire", ROOT / "2"),
    ("ds3", "FASDD_CV", ROOT / "3"),
    ("ds5", "FASD", ROOT / "5"),
]
MAX_SAMPLES = 3000
PATCH = 128
GLCM_DIST = [1, 2]
METRICS = [
    ("fft_band_low", "FFT Low"),
    ("fft_band_high", "FFT High"),
    ("dwt_energy_L0", "DWT L0"),
    ("dwt_energy_L1", "DWT L1"),
    ("glcm_contrast", "GLCM Contrast"),
    ("glcm_homogeneity", "GLCM Homogeneity"),
    ("edge_density_canny", "Edge Density"),
    ("sobel_mag_mean", "Sobel Mean"),
    ("laplacian_var", "Laplacian Var"),
]


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def load_class_rows(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        grouped = {"fire": [], "smoke": [], "background": []}
        for row in reader:
            grouped[row["class"]].append(row)
    return fields, grouped


def measure_background(roi: np.ndarray) -> dict:
    gray = preprocess_roi(roi, PATCH)
    _bins, prof = radial_profile(fft_log_magnitude(gray))
    lo, mid, hi = band_energies_from_profile(prof)
    energies, n_levels = wavelet_level_energies(gray, "db2", 4)
    row = {
        "patch_size": PATCH,
        "fft_band_low": lo,
        "fft_band_mid": mid,
        "fft_band_high": hi,
        "dwt_num_levels": n_levels,
    }
    for i, energy in enumerate(energies):
        row[f"dwt_energy_L{i}"] = energy
    row.update(glcm_properties(gray, GLCM_DIST))
    row.update(gradient_edge_features(gray, 50, 150))
    return row


def rerun_one(folder: str, name: str, root: Path) -> dict:
    fields, grouped = load_class_rows(OLD / folder / "multifeature_per_sample.csv")
    fire_rows = grouped["fire"]
    smoke_rows = grouped["smoke"]
    if len(fire_rows) != MAX_SAMPLES or len(smoke_rows) != MAX_SAMPLES:
        raise SystemExit(f"{name} expected 3000 fire/smoke, got {len(fire_rows)}/{len(smoke_rows)}")
    rng = random.Random(42)
    pairs = collect_image_label_pairs(root, splits=("train",))
    rng.shuffle(pairs)
    out_rows: list[dict] = []
    audit: list[dict] = []
    n_fire = n_smoke = n_bg = 0
    seen = 0
    no_patch = 0
    dataset_root = fire_rows[0]["dataset_root"]
    for img_path, lbl_path in pairs:
        if n_fire >= MAX_SAMPLES and n_smoke >= MAX_SAMPLES and n_bg >= MAX_SAMPLES:
            break
        seen += 1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        lines = [ln for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept = yolo_boxes_to_pixels(lines, img_w, img_h)
        for cls, x1, y1, x2, y2 in kept:
            if cls == 0 and n_fire >= MAX_SAMPLES:
                continue
            if cls == 1 and n_smoke >= MAX_SAMPLES:
                continue
            if cls not in (0, 1):
                continue
            tag = f"{img_path.stem}_cls{cls}_{x1}_{y1}"
            src = fire_rows if cls == 0 else smoke_rows
            idx = n_fire if cls == 0 else n_smoke
            if idx >= len(src) or src[idx]["tag"] != tag:
                published = src[idx]["tag"] if idx < len(src) else "END"
                raise SystemExit(f"{name} {('fire' if cls == 0 else 'smoke')} tag mismatch at {idx}: {tag} != {published}")
            out_rows.append(src[idx])
            if cls == 0:
                n_fire += 1
            else:
                n_smoke += 1
        if n_bg >= MAX_SAMPLES:
            continue
        every = yolo_all_boxes_xyxy(lines, img_w, img_h)
        patch = sample_zero_overlap_patch(img_w, img_h, every, rng, 48, 160)
        if patch is None:
            no_patch += 1
            continue
        if any(intersection_area(patch, box) > 0 for box in every):
            raise SystemExit(f"{name} overlap leaked: {img_path.name} {patch}")
        bx1, by1, bx2, by2 = patch
        roi = crop_safe(img, bx1, by1, bx2, by2)
        if roi.size == 0 or roi.shape[0] != by2 - by1 or roi.shape[1] != bx2 - bx1:
            raise SystemExit(f"{name} empty crop: {img_path.name} {patch}")
        measured = measure_background(roi)
        tag = f"{img_path.stem}_bg_{bx1}_{by1}"
        row = {
            "class": "background",
            "tag": tag,
            "local_index": n_bg,
            "image_stem": img_path.stem,
            "dataset_root": dataset_root,
        }
        row.update(measured)
        out_rows.append(row)
        audit.append(
            {
                "dataset": name,
                "tag": tag,
                "image_stem": img_path.stem,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "pw": bx2 - bx1,
                "ph": by2 - by1,
                "n_gt": len(every),
                "intersection_pixels": 0,
            }
        )
        n_bg += 1
        if n_bg % 250 == 0:
            print(f"{name} background {n_bg} after {seen} images, no-patch {no_patch}", flush=True)
    if n_bg != MAX_SAMPLES:
        raise SystemExit(f"{name} background {n_bg} < {MAX_SAMPLES} after {seen} images, no-patch {no_patch}")
    dest = OUT / folder
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "multifeature_per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    df = pd.read_csv(dest / "multifeature_per_sample.csv")
    num_cols = [c for c in df.columns if c not in ("class", "tag", "local_index", "image_stem", "dataset_root")]
    means = df.groupby("class")[num_cols].mean(numeric_only=True).reset_index()
    means.to_csv(dest / "multifeature_class_means.csv", index=False)
    fft_rows = []
    for cls, sub in df.groupby("class", sort=False):
        fft_rows.append(
            {
                "class": cls,
                "n_samples": int(len(sub)),
                "fft_size": PATCH,
                "band_low_radial_sum_mean": float(sub["fft_band_low"].mean()),
                "band_mid_radial_sum_mean": float(sub["fft_band_mid"].mean()),
                "band_high_radial_sum_mean": float(sub["fft_band_high"].mean()),
                "dataset_root": dataset_root,
            }
        )
    pd.DataFrame(fft_rows).to_csv(dest / "fft_energy_summary.csv", index=False)
    print(
        f"{name} done fire={n_fire} smoke={n_smoke} background={n_bg} "
        f"images={seen} no_patch={no_patch}",
        flush=True,
    )
    return {"name": name, "seen": seen, "no_patch": no_patch, "audit": audit, "df": df}


def order_of(means: dict[str, float]) -> str:
    rank = sorted(means, key=means.get, reverse=True)
    return ">".join({"fire": "F", "smoke": "S", "background": "B"}[c] for c in rank)


def write_comparison(results: list[dict]) -> None:
    lines = [
        "# ROI features recomputed with zero-overlap backgrounds",
        "",
        "Fire and smoke rows are identical to the original analysis (training split, seed 42, 3,000 per class).",
        "The 3,000 background patches (48–160 px) have zero pixel intersection with all annotated boxes, including boxes smaller than 20 px.",
        "",
        "| Dataset | Traversed images | Images without a valid patch | Backgrounds |",
        "|---|---:|---:|---:|",
    ]
    audits = []
    for result in results:
        lines.append(
            f"| {result['name']} | {result['seen']} | {result['no_patch']} | 3000 |"
        )
        audits.extend(result["audit"])
    lines.extend(["", "## Class rankings", "", "| Dataset | Metric | Original ranking | Zero-overlap ranking | Original background mean | Zero-overlap background mean |", "|---|---|---|---|---:|---:|"])
    for result in results:
        folder = {"D-Fire": "ds2", "FASDD_CV": "ds3", "FASD": "ds5"}[result["name"]]
        old = pd.read_csv(OLD / folder / "multifeature_class_means.csv")
        new = pd.read_csv(OUT / folder / "multifeature_class_means.csv")
        for col, label in METRICS:
            old_means = {row["class"]: float(row[col]) for _, row in old.iterrows()}
            new_means = {row["class"]: float(row[col]) for _, row in new.iterrows()}
            lines.append(
                f"| {result['name']} | {label} | {order_of(old_means)} | {order_of(new_means)} | "
                f"{old_means['background']:.6g} | {new_means['background']:.6g} |"
            )
    text = "\n".join(lines) + "\n"
    (OUT / "comparison_vs_iou005.md").write_text(text, encoding="utf-8")
    pd.DataFrame(audits).to_csv(OUT / "background_patches.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [rerun_one(folder, name, root) for folder, name, root in DATASETS]
    write_comparison(results)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
