#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""
Apply the same ROI analysis pipeline to YOLO-format fire/smoke datasets:
  1) FFT radial log-magnitude profile
  2) 2D DWT (PyWavelets)
  3) GLCM texture (scikit-image)
  4) Sobel / Canny edge and gradient features

Requirements:
  pip install opencv-python numpy matplotlib pandas pywavelets scikit-image
  (pandas is optional and only used for the summary tables)
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pywt  # PyWavelets
except ImportError:
    pywt = None  # type: ignore

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    graycomatrix = None  # type: ignore
    graycoprops = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore


def yolo_boxes_to_pixels(
    lines: List[str], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int, int]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        px_w = bw * img_w
        px_h = bh * img_h
        if px_w < 20 or px_h < 20:
            continue
        cx = xc * img_w
        cy = yc * img_h
        x1 = int(round(cx - px_w / 2))
        y1 = int(round(cy - px_h / 2))
        x2 = int(round(cx + px_w / 2))
        y2 = int(round(cy + px_h / 2))
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def box_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou_with_boxes(
    patch: Tuple[int, int, int, int],
    object_boxes: Sequence[Tuple[int, int, int, int]],
) -> float:
    if not object_boxes:
        return 0.0
    return float(max(box_iou_xyxy(patch, ob) for ob in object_boxes))


def try_random_background_patch(
    img_w: int,
    img_h: int,
    patch_w: int,
    patch_h: int,
    object_boxes: Sequence[Tuple[int, int, int, int]],
    max_iou: float,
    rng: random.Random,
    max_attempts: int = 500,
) -> Optional[Tuple[int, int, int, int]]:
    if patch_w >= img_w or patch_h >= img_h:
        return None
    for _ in range(max_attempts):
        x1 = rng.randint(0, img_w - patch_w)
        y1 = rng.randint(0, img_h - patch_h)
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        patch = (x1, y1, x2, y2)
        if max_iou_with_boxes(patch, object_boxes) <= max_iou:
            return patch
    return None


def yolo_all_boxes_xyxy(
    lines: List[str], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    """For background selection: all GT boxes (including those < 20 px) as half-open pixel boxes."""
    boxes: List[Tuple[int, int, int, int]] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        px_w = bw * img_w
        px_h = bh * img_h
        cx = xc * img_w
        cy = yc * img_h
        x1 = int(round(cx - px_w / 2))
        y1 = int(round(cy - px_h / 2))
        x2 = int(round(cx + px_w / 2))
        y2 = int(round(cy + px_h / 2))
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def _window_sum(ii: np.ndarray, x: int, y: int, pw: int, ph: int) -> float:
    return float(ii[y + ph, x + pw] - ii[y, x + pw] - ii[y + ph, x] + ii[y, x])


def _any_free_window(
    ii: np.ndarray,
    pw: int,
    ph: int,
    rng: random.Random,
    probes: int,
    exhaustive: bool,
) -> Optional[Tuple[int, int]]:
    """Top-left corner of a pw×ph window whose integral-image sum is zero, or None."""
    height = ii.shape[0] - 1
    width = ii.shape[1] - 1
    max_x = width - pw
    max_y = height - ph
    if max_x < 0 or max_y < 0:
        return None
    for _ in range(probes):
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        if _window_sum(ii, x, y, pw, ph) == 0.0:
            return x, y
    if not exhaustive:
        return None
    sums = ii[ph:, pw:] - ii[:-ph, pw:] - ii[ph:, :-pw] + ii[:-ph, :-pw]
    ys, xs = np.nonzero(sums == 0)
    if len(xs) == 0:
        return None
    pick = rng.randrange(len(xs))
    return int(xs[pick]), int(ys[pick])


def sample_zero_overlap_patch(
    img_w: int,
    img_h: int,
    object_boxes: Sequence[Tuple[int, int, int, int]],
    rng: random.Random,
    bg_min: int = 48,
    bg_max: int = 160,
    size_tries: int = 40,
) -> Optional[Tuple[int, int, int, int]]:
    """Background patch with zero pixel intersection with all GT boxes.

    Side lengths are drawn from [bg_min, min(bg_max, image side - 1)].
    If random sizes keep failing, sizes on a 4-pixel grid that actually fit are tried.
    The grid includes bg_min, so an image is skipped only if no empty square of side bg_min exists.
    """
    if img_w <= bg_min or img_h <= bg_min:
        return None
    pw_hi = min(bg_max, img_w - 1)
    ph_hi = min(bg_max, img_h - 1)
    if pw_hi < bg_min or ph_hi < bg_min:
        return None
    occ = np.zeros((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2 in object_boxes:
        occ[y1:y2, x1:x2] = 1
    ii = cv2.integral(occ)
    for _ in range(size_tries):
        pw = rng.randint(bg_min, pw_hi)
        ph = rng.randint(bg_min, ph_hi)
        origin = _any_free_window(ii, pw, ph, rng, probes=250, exhaustive=False)
        if origin is not None:
            x, y = origin
            return x, y, x + pw, y + ph
    sizes = [
        (pw, ph)
        for pw in range(bg_min, pw_hi + 1, 4)
        for ph in range(bg_min, ph_hi + 1, 4)
    ]
    rng.shuffle(sizes)
    for pw, ph in sizes:
        origin = _any_free_window(ii, pw, ph, rng, probes=40, exhaustive=True)
        if origin is not None:
            x, y = origin
            return x, y, x + pw, y + ph
    return None


def crop_safe(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return img[y1:y2, x1:x2].copy()


def preprocess_roi(roi_bgr: np.ndarray, patch_size: int) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    g = g - np.mean(g)
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    return np.log1p(mag)


def radial_profile(log_mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = log_mag.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    vals = log_mag.ravel()
    max_r = int(np.ceil(rr.max())) + 1
    sum_w = np.bincount(np.minimum(rr.astype(np.int64), max_r - 1), weights=vals, minlength=max_r)
    cnt = np.bincount(np.minimum(rr.astype(np.int64), max_r - 1), minlength=max_r)
    prof = sum_w / np.maximum(cnt, 1)
    return np.arange(max_r), prof


def band_energies_from_profile(profile: np.ndarray) -> Tuple[float, float, float]:
    """Sum of the radial profile over three equal radial intervals (Low/Mid/High)."""
    L = len(profile)
    if L <= 0:
        return float("nan"), float("nan"), float("nan")
    r1 = max(1, L // 3)
    r2 = max(r1 + 1, 2 * L // 3)
    return float(np.sum(profile[:r1])), float(np.sum(profile[r1:r2])), float(np.sum(profile[r2:]))


def wavelet_level_energies(gray: np.ndarray, wavelet: str, dwt_level: int) -> Tuple[List[float], int]:
    """Energy (sum of squares) per wavedec2 level. coeffs[0] = approximation, others = details."""
    if pywt is None:
        return [], 0
    g = gray.astype(np.float64) / 255.0
    level = dwt_level
    coeffs = pywt.wavedec2(g, wavelet, mode="reflect", level=level)
    energies: List[float] = []
    c0 = coeffs[0]
    energies.append(float(np.sum(c0**2)))
    for tup in coeffs[1:]:
        e = sum(float(np.sum(c**2)) for c in tup)
        energies.append(e)
    return energies, len(energies)


def glcm_properties(gray: np.ndarray, distances: List[int]) -> Dict[str, float]:
    """GLCM properties averaged over angles."""
    if graycomatrix is None or graycoprops is None:
        return {
            "glcm_contrast": float("nan"),
            "glcm_homogeneity": float("nan"),
            "glcm_energy": float("nan"),
            "glcm_correlation": float("nan"),
            "glcm_asm": float("nan"),
        }
    g_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    M = graycomatrix(
        g_u8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(M, "contrast").mean()
    homogeneity = graycoprops(M, "homogeneity").mean()
    energy = graycoprops(M, "energy").mean()
    correlation = graycoprops(M, "correlation").mean()
    asm = graycoprops(M, "ASM").mean()
    return {
        "glcm_contrast": float(contrast),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
        "glcm_asm": float(asm),
    }


def gradient_edge_features(gray: np.ndarray, canny_lo: int, canny_hi: int) -> Dict[str, float]:
    g = gray.astype(np.float64)
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, canny_lo, canny_hi)
    edge_density = float(np.mean(edges > 0))
    lap = cv2.Laplacian(u8, cv2.CV_64F)
    lap_var = float(lap.var())
    return {
        "sobel_mag_mean": float(mag.mean()),
        "sobel_mag_std": float(mag.std()),
        "edge_density_canny": edge_density,
        "laplacian_var": lap_var,
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def collect_image_label_pairs(
    root: Path, splits: Sequence[str] = ("train", "val", "test")
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"):
            for img_path in img_dir.glob(f"*{ext}"):
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.is_file():
                    pairs.append((img_path, lbl_path))
    return pairs


def save_spectrum_png(path: Path, log_mag: np.ndarray) -> None:
    m = log_mag.astype(np.float64)
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    im = (m * 255).astype(np.uint8)
    cv2.imwrite(str(path), im)


def stack_profiles(profiles: List[np.ndarray]) -> Optional[np.ndarray]:
    if not profiles:
        return None
    max_len = max(len(p) for p in profiles)
    padded = []
    for p in profiles:
        pp = np.pad(p, (0, max_len - len(p)), mode="edge") if len(p) < max_len else p[:max_len]
        padded.append(pp)
    return np.stack(padded, axis=0)


def append_csv_row(path: Path, fieldnames: List[str], row: Dict, write_header: bool) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO ROI: FFT + DWT + GLCM + Gradient/Edge multifeature analysis"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="YOLO-format dataset root (images/, labels/)")
    parser.add_argument("--output-dir", type=Path, default=Path("./multifeature_analysis_results"))
    parser.add_argument("--patch-size", type=int, default=128, choices=[128, 256])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="splits to analyze (comma-separated), e.g. train",
    )
    parser.add_argument("--bg-patch-min", type=int, default=48)
    parser.add_argument("--bg-patch-max", type=int, default=160)
    parser.add_argument("--bg-max-iou", type=float, default=0.05)
    parser.add_argument(
        "--bg-zero-overlap",
        action="store_true",
        help="use only background patches with zero pixel intersection with all GT boxes (incl. < 20 px)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", type=str, default="db2", help="PyWavelets wavelet name")
    parser.add_argument("--dwt-level", type=int, default=4, help="DWT decomposition level")
    parser.add_argument("--glcm-distances", type=str, default="1,2", help="GLCM distances (comma-separated)")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument(
        "--no-save-per-sample-fft-png",
        action="store_true",
        help="skip per-sample FFT PNGs (mean images are kept)",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="skip per-sample ROI crop PNGs",
    )
    args = parser.parse_args()

    if pywt is None:
        print("[ERROR] PyWavelets is required: pip install PyWavelets", file=sys.stderr)
        sys.exit(1)
    if graycomatrix is None or graycoprops is None:
        print("[ERROR] scikit-image is required: pip install scikit-image", file=sys.stderr)
        sys.exit(1)

    glcm_dist = [int(x.strip()) for x in args.glcm_distances.split(",") if x.strip()]

    rng = random.Random(args.seed)
    root = args.dataset_root.resolve()
    out = args.output_dir.resolve()
    ps = args.patch_size

    crops_fire = out / "crops" / "fire"
    crops_smoke = out / "crops" / "smoke"
    crops_bg = out / "crops" / "background"
    fft_fire = out / "fft" / "fire"
    fft_smoke = out / "fft" / "smoke"
    fft_bg = out / "fft" / "background"
    for d in (crops_fire, crops_smoke, crops_bg, fft_fire, fft_smoke, fft_bg):
        ensure_dir(d)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    pairs = collect_image_label_pairs(root, splits=splits)
    rng.shuffle(pairs)

    n_fire = n_smoke = n_bg = 0
    fire_mags: List[np.ndarray] = []
    smoke_mags: List[np.ndarray] = []
    bg_mags: List[np.ndarray] = []
    fire_profiles: List[np.ndarray] = []
    smoke_profiles: List[np.ndarray] = []
    bg_profiles: List[np.ndarray] = []

    per_sample_rows: List[Dict] = []
    idx_fire = idx_smoke = idx_bg = 0

    for img_path, lbl_path in pairs:
        if n_fire >= args.max_samples and n_smoke >= args.max_samples and n_bg >= args.max_samples:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        with open(lbl_path) as f:
            lines = [ln for ln in f.readlines() if ln.strip()]

        all_pixel_boxes = yolo_boxes_to_pixels(lines, img_w, img_h)
        object_xyxy = [(b[1], b[2], b[3], b[4]) for b in all_pixel_boxes]

        def process_roi(roi: np.ndarray, cls_name: str, tag: str, local_idx: int) -> None:
            gray = preprocess_roi(roi, ps)
            log_mag = fft_log_magnitude(gray)
            _, prof = radial_profile(log_mag)
            lo, mid, hi = band_energies_from_profile(prof)

            w_energies, n_wlev = wavelet_level_energies(gray, args.wavelet, args.dwt_level)
            glcm = glcm_properties(gray, glcm_dist)
            ge = gradient_edge_features(gray, args.canny_low, args.canny_high)

            row: Dict = {
                "class": cls_name,
                "tag": tag,
                "local_index": local_idx,
                "image_stem": img_path.stem,
                "patch_size": ps,
                "fft_band_low": lo,
                "fft_band_mid": mid,
                "fft_band_high": hi,
                "dwt_num_levels": n_wlev,
                "dataset_root": str(root),
            }
            for i, e in enumerate(w_energies):
                row[f"dwt_energy_L{i}"] = e
            row.update(glcm)
            row.update(ge)
            per_sample_rows.append(row)

        for cls, x1, y1, x2, y2 in all_pixel_boxes:
            if cls == 0 and n_fire >= args.max_samples:
                continue
            if cls == 1 and n_smoke >= args.max_samples:
                continue
            if cls not in (0, 1):
                continue

            roi = crop_safe(img, x1, y1, x2, y2)
            if roi.size == 0:
                continue
            gray = preprocess_roi(roi, ps)
            log_mag = fft_log_magnitude(gray)
            tag = f"{img_path.stem}_cls{cls}_{x1}_{y1}"
            cls_name = "fire" if cls == 0 else "smoke"

            if cls == 0:
                if not args.no_save_crops:
                    cv2.imwrite(str(crops_fire / f"{idx_fire:05d}_{tag}.png"), roi)
                if not args.no_save_per_sample_fft_png:
                    save_spectrum_png(fft_fire / f"{idx_fire:05d}_{tag}.png", log_mag)
                fire_mags.append(log_mag)
                _, p = radial_profile(log_mag)
                fire_profiles.append(p)
                process_roi(roi, "fire", tag, idx_fire)
                idx_fire += 1
                n_fire += 1
            else:
                if not args.no_save_crops:
                    cv2.imwrite(str(crops_smoke / f"{idx_smoke:05d}_{tag}.png"), roi)
                if not args.no_save_per_sample_fft_png:
                    save_spectrum_png(fft_smoke / f"{idx_smoke:05d}_{tag}.png", log_mag)
                smoke_mags.append(log_mag)
                _, p = radial_profile(log_mag)
                smoke_profiles.append(p)
                process_roi(roi, "smoke", tag, idx_smoke)
                idx_smoke += 1
                n_smoke += 1

        if n_bg < args.max_samples:
            patch = None
            if args.bg_zero_overlap:
                every_xyxy = yolo_all_boxes_xyxy(lines, img_w, img_h)
                patch = sample_zero_overlap_patch(
                    img_w,
                    img_h,
                    every_xyxy,
                    rng,
                    bg_min=args.bg_patch_min,
                    bg_max=args.bg_patch_max,
                )
            elif img_w > args.bg_patch_min and img_h > args.bg_patch_min:
                pw_hi = min(args.bg_patch_max, img_w - 1)
                ph_hi = min(args.bg_patch_max, img_h - 1)
                if pw_hi >= args.bg_patch_min and ph_hi >= args.bg_patch_min:
                    pw = rng.randint(args.bg_patch_min, pw_hi)
                    ph = rng.randint(args.bg_patch_min, ph_hi)
                    patch = try_random_background_patch(
                        img_w, img_h, pw, ph, object_xyxy, args.bg_max_iou, rng
                    )
            if patch is not None:
                bx1, by1, bx2, by2 = patch
                roi = crop_safe(img, bx1, by1, bx2, by2)
                if roi.size > 0:
                    gray = preprocess_roi(roi, ps)
                    log_mag = fft_log_magnitude(gray)
                    tag = f"{img_path.stem}_bg_{bx1}_{by1}"
                    if not args.no_save_crops:
                        cv2.imwrite(str(crops_bg / f"{idx_bg:05d}_{tag}.png"), roi)
                    if not args.no_save_per_sample_fft_png:
                        save_spectrum_png(fft_bg / f"{idx_bg:05d}_{tag}.png", log_mag)
                    bg_mags.append(log_mag)
                    _, p = radial_profile(log_mag)
                    bg_profiles.append(p)
                    process_roi(roi, "background", tag, idx_bg)
                    idx_bg += 1
                    n_bg += 1

    # mean FFT PNGs
    if fire_mags:
        _save_avg(out / "avg_fft_fire.png", fire_mags)
    if smoke_mags:
        _save_avg(out / "avg_fft_smoke.png", smoke_mags)
    if bg_mags:
        _save_avg(out / "avg_fft_background.png", bg_mags)

    pf, ps_, pb = stack_profiles(fire_profiles), stack_profiles(smoke_profiles), stack_profiles(bg_profiles)
    _plot_radial(out / "radial_profile_comparison.png", pf, ps_, pb)

    # fft_energy_summary.csv
    fft_rows = []
    for name, stack in (("fire", pf), ("smoke", ps_), ("background", pb)):
        if stack is None or stack.size == 0:
            fft_rows.append(
                {
                    "class": name,
                    "n_samples": 0,
                    "fft_size": ps,
                    "band_low_radial_sum_mean": float("nan"),
                    "band_mid_radial_sum_mean": float("nan"),
                    "band_high_radial_sum_mean": float("nan"),
                    "dataset_root": str(root),
                }
            )
            continue
        lows, mids, highs = [], [], []
        for i in range(stack.shape[0]):
            p = stack[i]
            lo, mid, hi = band_energies_from_profile(p)
            lows.append(lo)
            mids.append(mid)
            highs.append(hi)
        fft_rows.append(
            {
                "class": name,
                "n_samples": int(stack.shape[0]),
                "fft_size": ps,
                "band_low_radial_sum_mean": float(np.mean(lows)),
                "band_mid_radial_sum_mean": float(np.mean(mids)),
                "band_high_radial_sum_mean": float(np.mean(highs)),
                "dataset_root": str(root),
            }
        )
    if pd is not None:
        pd.DataFrame(fft_rows).to_csv(out / "fft_energy_summary.csv", index=False)
    else:
        with open(out / "fft_energy_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "class",
                    "n_samples",
                    "fft_size",
                    "band_low_radial_sum_mean",
                    "band_mid_radial_sum_mean",
                    "band_high_radial_sum_mean",
                    "dataset_root",
                ],
            )
            w.writeheader()
            w.writerows(fft_rows)

    # per-sample CSV
    if per_sample_rows:
        dwt_cols = max((r.get("dwt_num_levels", 0) or 0) for r in per_sample_rows)
        base_fields = [
            "class",
            "tag",
            "local_index",
            "image_stem",
            "patch_size",
            "fft_band_low",
            "fft_band_mid",
            "fft_band_high",
            "dwt_num_levels",
        ]
        dwt_fields = [f"dwt_energy_L{i}" for i in range(dwt_cols)]
        tail = [
            "glcm_contrast",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "glcm_asm",
            "sobel_mag_mean",
            "sobel_mag_std",
            "edge_density_canny",
            "laplacian_var",
            "dataset_root",
        ]
        all_fields = base_fields + dwt_fields + tail
        sample_path = out / "multifeature_per_sample.csv"
        with open(sample_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            w.writeheader()
            for r in per_sample_rows:
                for i in range(dwt_cols):
                    if f"dwt_energy_L{i}" not in r:
                        r[f"dwt_energy_L{i}"] = ""
                w.writerow(r)

    # class-wise means (numeric columns only)
    if per_sample_rows and pd is not None:
        df = pd.DataFrame(per_sample_rows)
        num_cols = [c for c in df.columns if c not in ("class", "tag", "local_index", "image_stem", "dataset_root")]
        agg = df.groupby("class")[num_cols].mean(numeric_only=True).reset_index()
        agg.to_csv(out / "multifeature_class_means.csv", index=False)

    _plot_comparison_bars(out / "comparison_bar_wavelet_approx.png", per_sample_rows, "dwt_energy_L0", "Wavelet approx energy")
    _plot_comparison_bars(out / "comparison_bar_glcm_contrast.png", per_sample_rows, "glcm_contrast", "GLCM contrast (mean)")
    _plot_comparison_bars(out / "comparison_bar_sobel.png", per_sample_rows, "sobel_mag_mean", "Sobel magnitude mean")

    print(f"done: {out}")
    print(f"  splits={splits}")
    print(f"  fire={n_fire}, smoke={n_smoke}, background={n_bg} (max {args.max_samples})")
    print("  outputs: fft_energy_summary.csv, multifeature_per_sample.csv, multifeature_class_means.csv (with pandas)")


def _save_avg(path: Path, mags: List[np.ndarray]) -> None:
    avg = np.mean(np.stack(mags, axis=0), axis=0)
    save_spectrum_png(path, avg)


def _plot_radial(
    path: Path,
    pf: Optional[np.ndarray],
    ps: Optional[np.ndarray],
    pb: Optional[np.ndarray],
) -> None:
    plt.figure(figsize=(8, 5))
    for stack, label, color in (
        (pf, "fire", "tab:red"),
        (ps, "smoke", "tab:blue"),
        (pb, "background", "tab:gray"),
    ):
        if stack is None:
            continue
        m = stack.mean(axis=0)
        s = stack.std(axis=0)
        x = np.arange(len(m))
        plt.plot(x, m, label=label, color=color)
        plt.fill_between(x, m - s, m + s, alpha=0.2, color=color)
    plt.xlabel("Radial bin")
    plt.ylabel("Mean log(1 + |FFT|)")
    plt.title("Radial FFT profile (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_comparison_bars(path: Path, rows: List[Dict], key: str, title: str) -> None:
    if not rows:
        return
    by_c: Dict[str, List[float]] = {"fire": [], "smoke": [], "background": []}
    for r in rows:
        c = r.get("class")
        if c not in by_c:
            continue
        v = r.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            by_c[c].append(float(v))
        except (TypeError, ValueError):
            continue
    means = {c: (float(np.mean(v)) if v else float("nan")) for c, v in by_c.items()}
    classes = [c for c in ("fire", "smoke", "background") if not np.isnan(means[c])]
    if not classes:
        return
    vals = [means[c] for c in classes]
    plt.figure(figsize=(6, 4))
    plt.bar(classes, vals, color=["tab:red", "tab:blue", "tab:gray"][: len(classes)])
    plt.title(title)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
