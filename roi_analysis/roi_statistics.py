#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""Statistical stability of ROI class differences (Section 2.3, Table 8).

From the existing multifeature_per_sample.csv files (27,000 ROIs), this script computes the
sample structure, descriptive statistics, standardized effect sizes (Cohen's d and Hedges' g),
and image-level cluster-bootstrap 95% CIs of the class-mean differences, and draws the
corresponding figures. ROIs are not re-extracted.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Root folder that contains the ROI feature folders produced by roi_multifeature_analysis.py.
# Can be overridden with --root or the ROI_DATA_ROOT environment variable.
ROOT = Path(os.environ.get("ROI_DATA_ROOT", "./data"))
SRC = ROOT / "multifeature_train3000_235"
OUT = ROOT / "roi_validation_32"

DATASETS = [
    ("ds2", "D-Fire", 2),
    ("ds3", "FASDD_CV", 3),
    ("ds5", "FASD", 5),
]
CLASS_ORDER = ["fire", "smoke", "background"]
CLASS_LABEL = {"fire": "Fire", "smoke": "Smoke", "background": "Background"}
CLASS_COLOR = {"fire": "#d97706", "smoke": "#0d9488", "background": "#6366f1"}

# Paper notation -> CSV column. L0 = wavedec2 approximation (A4), L1 = coarsest detail level (D4).
METRICS = {
    "sobel_mean": ("sobel_mag_mean", "Sobel Mean"),
    "edge_density": ("edge_density_canny", "Edge Density"),
    "dwt_a4": ("dwt_energy_L0", "DWT A4"),
    "dwt_d4": ("dwt_energy_L1", "DWT D4"),
    "fft_low": ("fft_band_low", "FFT Low"),
    "fft_high": ("fft_band_high", "FFT High"),
    "glcm_contrast": ("glcm_contrast", "GLCM Contrast"),
    "glcm_homogeneity": ("glcm_homogeneity", "GLCM Homogeneity"),
    "laplacian_var": ("laplacian_var", "Laplacian Var"),
}

# (metric_key, class_a, class_b, priority)
COMPARISONS = [
    ("sobel_mean", "fire", "smoke", "primary"),
    ("sobel_mean", "fire", "background", "primary"),
    ("sobel_mean", "smoke", "background", "primary"),
    ("dwt_a4", "smoke", "fire", "primary"),
    ("dwt_a4", "smoke", "background", "primary"),
    ("dwt_d4", "fire", "smoke", "primary"),
    ("dwt_d4", "fire", "background", "primary"),
    ("edge_density", "fire", "smoke", "secondary"),
    ("edge_density", "fire", "background", "secondary"),
    ("fft_low", "fire", "smoke", "secondary"),
    ("fft_high", "smoke", "fire", "secondary"),
    ("fft_high", "smoke", "background", "secondary"),
]

N_BOOT = 2000
SEED = 42
EXPECTED_N = 3000


def load_frames() -> pd.DataFrame:
    frames = []
    for folder, name, ds_id in DATASETS:
        path = SRC / folder / "multifeature_per_sample.csv"
        part = pd.read_csv(path)
        part["dataset"] = name
        part["dataset_id"] = folder
        part["folder_id"] = ds_id
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["class"] = df["class"].str.strip().str.lower()
    return df


def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


def audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("dataset", sort=False):
        class_sets = {
            stem: set(sub["class"])
            for stem, sub in g.groupby("image_stem")
        }
        n_images_fire_smoke = sum(
            1 for classes in class_sets.values() if {"fire", "smoke"} <= classes
        )
        for cls in CLASS_ORDER:
            gc = g[g["class"] == cls]
            per_image = gc.groupby("image_stem").size()
            rows.append(
                {
                    "dataset": name,
                    "class": CLASS_LABEL[cls],
                    "n_roi": int(len(gc)),
                    "n_unique_tag": int(gc["tag"].nunique()),
                    "n_duplicate_tag": int(gc["tag"].duplicated().sum()),
                    "n_unique_image": int(gc["image_stem"].nunique()),
                    "roi_per_image_mean": float(per_image.mean()) if len(per_image) else np.nan,
                    "roi_per_image_max": int(per_image.max()) if len(per_image) else 0,
                    "n_images_with_2plus_roi": int((per_image >= 2).sum()),
                    "n_images_with_fire_and_smoke": n_images_fire_smoke,
                    "n_rows_expected": EXPECTED_N,
                    "count_ok": int(len(gc) == EXPECTED_N and gc["tag"].duplicated().sum() == 0),
                }
            )
    return pd.DataFrame(rows)


def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("dataset", sort=False):
        for key, (col, label) in METRICS.items():
            means = {}
            for cls in CLASS_ORDER:
                s = g.loc[g["class"] == cls, col].dropna()
                means[cls] = float(s.mean())
                rows.append(
                    {
                        "dataset": name,
                        "metric": label,
                        "metric_key": key,
                        "class": CLASS_LABEL[cls],
                        "n": int(s.shape[0]),
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=1)),
                        "median": float(s.median()),
                        "q1": float(s.quantile(0.25)),
                        "q3": float(s.quantile(0.75)),
                        "iqr": iqr(s),
                    }
                )
            order = sorted(CLASS_ORDER, key=lambda c: means[c], reverse=True)
            rank = ">".join(CLASS_LABEL[c][0] for c in order)
            for row in rows[-3:]:
                row["mean_rank"] = rank
    return pd.DataFrame(rows)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g: pooled-SD Cohen's d multiplied by the exact small-sample correction J(df)."""
    na, nb = a.size, b.size
    df = na + nb - 2
    sa, sb = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = math.sqrt(((na - 1) * sa * sa + (nb - 1) * sb * sb) / df)
    if pooled == 0:
        return float("nan")
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    j = math.exp(math.lgamma(df / 2) - 0.5 * math.log(df / 2) - math.lgamma((df - 1) / 2))
    return j * d


def cluster_bootstrap(
    data: pd.DataFrame,
    metric_col: str,
    class_a: str,
    class_b: str,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    sub = data.loc[data["class"].isin([class_a, class_b]), ["image_stem", "class", metric_col]]
    sub = sub.dropna()
    grouped = (
        sub.groupby(["image_stem", "class"])[metric_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    classes = [class_a, class_b]
    sums = (
        grouped.pivot(index="image_stem", columns="class", values="sum")
        .reindex(columns=classes)
        .fillna(0.0)
    )
    counts = (
        grouped.pivot(index="image_stem", columns="class", values="count")
        .reindex(index=sums.index, columns=classes)
        .fillna(0.0)
    )
    sum_array = sums.to_numpy(dtype=np.float64)
    count_array = counts.to_numpy(dtype=np.float64)
    total_count = count_array.sum(axis=0)
    means = sum_array.sum(axis=0) / total_count
    observed = float(means[0] - means[1])

    rng = np.random.default_rng(seed)
    n_groups = sum_array.shape[0]
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    sampled_sum = sum_array[idx].sum(axis=1)
    sampled_count = count_array[idx].sum(axis=1)
    valid = np.all(sampled_count > 0, axis=1)
    diffs = np.full(n_boot, np.nan)
    diffs[valid] = sampled_sum[valid, 0] / sampled_count[valid, 0] - sampled_sum[valid, 1] / sampled_count[valid, 1]
    ok = diffs[valid]
    low, high = np.percentile(ok, [2.5, 97.5])
    return {
        "n_groups": int(n_groups),
        "n_boot_valid": int(valid.sum()),
        "mean_a": float(means[0]),
        "mean_b": float(means[1]),
        "mean_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_0": bool(low > 0 or high < 0),
    }


def effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for folder, name, _ds_id in DATASETS:
        g = df[df["dataset"] == name]
        for key, class_a, class_b, priority in COMPARISONS:
            col, label = METRICS[key]
            a = g.loc[g["class"] == class_a, col].dropna().to_numpy()
            b = g.loc[g["class"] == class_b, col].dropna().to_numpy()
            boot = cluster_bootstrap(g, col, class_a, class_b)
            rows.append(
                {
                    "dataset": name,
                    "metric": label,
                    "metric_key": key,
                    "comparison": f"{CLASS_LABEL[class_a]}–{CLASS_LABEL[class_b]}",
                    "class_a": CLASS_LABEL[class_a],
                    "class_b": CLASS_LABEL[class_b],
                    "priority": priority,
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "mean_a": boot["mean_a"],
                    "mean_b": boot["mean_b"],
                    "std_a": float(np.std(a, ddof=1)),
                    "std_b": float(np.std(b, ddof=1)),
                    "mean_difference": boot["mean_difference"],
                    "effect_size_d": cohen_d(a, b),
                    "effect_size_g": hedges_g(a, b),
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "ci_excludes_0": boot["ci_excludes_0"],
                    "n_groups": boot["n_groups"],
                    "n_boot": N_BOOT,
                    "n_boot_valid": boot["n_boot_valid"],
                    "group_unit": "image_stem",
                    "seed": SEED,
                }
            )
    return pd.DataFrame(rows)


def fmt_num(metric: str, value: float) -> str:
    if metric == "Edge Density":
        return f"{value:.4f}"
    if metric in ("GLCM Homogeneity",):
        return f"{value:.4f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_tables(audit_df: pd.DataFrame, desc: pd.DataFrame, eff: pd.DataFrame) -> None:
    audit_df.to_csv(OUT / "sample_audit.csv", index=False)
    desc.to_csv(OUT / "descriptive_stats.csv", index=False)
    eff.to_csv(OUT / "effect_size_bootstrap.csv", index=False)

    primary = eff[eff["priority"] == "primary"].copy()
    secondary = eff[eff["priority"] == "secondary"].copy()
    show_cols = [
        "dataset",
        "metric",
        "comparison",
        "mean_difference",
        "effect_size_g",
        "ci_low",
        "ci_high",
        "ci_excludes_0",
        "n_groups",
    ]
    primary[show_cols].to_csv(OUT / "table_3_3.csv", index=False)
    secondary[show_cols].to_csv(OUT / "table_3_3_supplement.csv", index=False)


def plot_distributions(df: pd.DataFrame) -> None:
    panels = [
        ("sobel_mag_mean", "Sobel Mean"),
        ("dwt_energy_L0", "DWT A4"),
        ("dwt_energy_L1", "DWT D4"),
        ("fft_band_high", "FFT High"),
    ]
    dataset_names = [name for _folder, name, _i in DATASETS]
    fig, axes = plt.subplots(len(panels), len(dataset_names), figsize=(11.5, 13.2), squeeze=False)
    for i, (col, title) in enumerate(panels):
        for j, name in enumerate(dataset_names):
            ax = axes[i, j]
            sub = df[df["dataset"] == name]
            data = [sub.loc[sub["class"] == cls, col].dropna().to_numpy() for cls in CLASS_ORDER]
            lo = min(np.quantile(v, 0.005) for v in data)
            hi = max(np.quantile(v, 0.995) for v in data)
            pad = (hi - lo) * 0.06
            parts = ax.violinplot(data, positions=[1, 2, 3], widths=0.78, showextrema=False)
            for body, cls in zip(parts["bodies"], CLASS_ORDER):
                body.set_facecolor(CLASS_COLOR[cls])
                body.set_edgecolor(CLASS_COLOR[cls])
                body.set_alpha(0.28)
            ax.boxplot(
                data,
                positions=[1, 2, 3],
                widths=0.16,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.2},
                boxprops={"color": "#333333", "linewidth": 0.8},
                whiskerprops={"color": "#333333", "linewidth": 0.8},
                capprops={"color": "#333333", "linewidth": 0.8},
            )
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels([CLASS_LABEL[c] for c in CLASS_ORDER], fontsize=8)
            if i == 0:
                ax.set_title(name, fontsize=12)
            if j == 0:
                ax.set_ylabel(title, fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Figure 3-4. ROI feature distributions by dataset", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "figure_3_4_distributions.png", dpi=160)
    fig.savefig(OUT / "figure_3_4_distributions.pdf")
    plt.close(fig)


def plot_ci(eff: pd.DataFrame) -> None:
    primary = eff[eff["priority"] == "primary"].copy()
    metrics = ["Sobel Mean", "DWT A4", "DWT D4"]
    dataset_names = [name for _folder, name, _i in DATASETS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.4, 6.2), sharey=False)
    for ax, metric in zip(axes, metrics):
        sub = primary[primary["metric"] == metric]
        ylabels = []
        y = 0
        ys = []
        for comparison in sub["comparison"].unique():
            for name in dataset_names:
                row = sub[(sub["comparison"] == comparison) & (sub["dataset"] == name)].iloc[0]
                color = "#b91c1c" if not row["ci_excludes_0"] else "#1f2937"
                ax.errorbar(
                    row["mean_difference"],
                    y,
                    xerr=[[row["mean_difference"] - row["ci_low"]], [row["ci_high"] - row["mean_difference"]]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.4,
                    capsize=3,
                    markersize=4.5,
                )
                ys.append(y)
                ylabels.append(f"{name}  {comparison}")
                y += 1
            y += 0.45
        ax.axvline(0, color="#9ca3af", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Mean difference (A − B)")
        ax.invert_yaxis()
    fig.suptitle("Image-level cluster bootstrap 95% CI  (n=2000, seed=42)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figure_3_4_bootstrap_ci.png", dpi=160)
    fig.savefig(OUT / "figure_3_4_bootstrap_ci.pdf")
    plt.close(fig)


def markdown_table(eff: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Metric | Comparison | Mean difference | Hedges g | 95% CI | Excludes 0 | Images |",
        "|---|---|---|---:|---:|---|---|---:|",
    ]
    for _, r in eff.iterrows():
        ci = f"[{fmt_num(r['metric'], r['ci_low'])}, {fmt_num(r['metric'], r['ci_high'])}]"
        lines.append(
            "| {dataset} | {metric} | {comparison} | {diff} | {d:.3f} | {ci} | {flag} | {n} |".format(
                dataset=r["dataset"],
                metric=r["metric"],
                comparison=r["comparison"],
                diff=fmt_num(r["metric"], r["mean_difference"]),
                d=r["effect_size_g"],
                ci=ci,
                flag="yes" if r["ci_excludes_0"] else "no",
                n=int(r["n_groups"]),
            )
        )
    return "\n".join(lines)


def sentence_block(eff: pd.DataFrame) -> str:
    def one(metric: str, comparison: str) -> str:
        bits = []
        for _, r in eff[(eff["metric"] == metric) & (eff["comparison"] == comparison)].iterrows():
            ci = f"[{fmt_num(metric, r['ci_low'])}, {fmt_num(metric, r['ci_high'])}]"
            bits.append(
                f"{r['dataset']} {fmt_num(metric, r['mean_difference'])} (95% CI {ci}, g={r['effect_size_g']:.2f})"
            )
        return "; ".join(bits)

    sobel = one("Sobel Mean", "Fire–Smoke")
    a4 = one("DWT A4", "Smoke–Fire")
    d4 = one("DWT D4", "Fire–Smoke")
    return "\n".join(
        [
            "Statistical check of the ROI analysis",
            "",
            "An image-level cluster bootstrap (resampling unit: original image identifier, 2,000 resamples) was applied within each dataset to obtain 95% confidence intervals of the class-mean differences.",
            "",
            f"Sobel Mean, Fire–Smoke: {sobel}.",
            "",
            f"DWT A4, Smoke–Fire: {a4}. DWT D4, Fire–Smoke: {d4}.",
            "",
            "These statistics describe input-image ROIs and are not causal evidence for detection-accuracy changes of FEPF or MDDM.",
        ]
    )


def write_report(df: pd.DataFrame, audit_df: pd.DataFrame, desc: pd.DataFrame, eff: pd.DataFrame) -> None:
    total = len(df)
    dup = int(df.duplicated(["dataset", "tag"]).sum())
    lines = [
        "# ROI statistics report",
        "",
        "Input: the existing `multifeature_per_sample.csv` files (training split, 3,000 ROIs per class and dataset, seed 42, 128×128 grayscale).",
        "",
        "- Input folders: `multifeature_train3000_235/ds2`, `ds3`, `ds5`",
        "- Mapping: ds2 = D-Fire, ds3 = FASDD_CV, ds5 = FASD",
        "- DWT A4 = `dwt_energy_L0` (level-4 approximation energy)",
        "- DWT D4 = `dwt_energy_L1` (coarsest detail level returned by `wavedec2`)",
        "- Cluster unit: original image identifier (no video IDs are available).",
        "- Bootstrap: 2,000 resamples, seed 42, percentile 95% CI",
        "- Effect size: ROI-level Hedges' g = J(df) × d, with d = (mean A − mean B) / pooled SD",
        "",
        "## 1. Sample structure",
        "",
        f"- Total rows: {total:,} (expected 27,000)",
        f"- Duplicated tags within a dataset: {dup}",
        "",
        "| Dataset | Class | ROIs | Unique images | Mean ROIs/image | Max | Images with ≥2 ROIs | Images with fire and smoke |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in audit_df.iterrows():
        lines.append(
            f"| {r['dataset']} | {r['class']} | {r['n_roi']} | {r['n_unique_image']} | {r['roi_per_image_mean']:.2f} | {r['roi_per_image_max']} | {r['n_images_with_2plus_roi']} | {r['n_images_with_fire_and_smoke']} |"
        )
    lines += [
        "",
        "The number of images containing both fire and smoke is a dataset-level value repeated in each class row.",
        "",
        "## 2. Class-mean rankings",
        "",
        "| Dataset | Metric | Ranking |",
        "|---|---|---|",
    ]
    rank = desc.drop_duplicates(["dataset", "metric"])
    for _, r in rank.iterrows():
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['mean_rank']} |")
    lines += [
        "",
        "## 3. Primary metrics (cluster bootstrap)",
        "",
        markdown_table(eff[eff["priority"] == "primary"]),
        "",
        "## 4. Secondary metrics",
        "",
        markdown_table(eff[eff["priority"] == "secondary"]),
        "",
        "## 5. Figures",
        "",
        "- `figure_3_4_distributions.png`: per-dataset distributions of Sobel Mean, DWT A4, DWT D4, and FFT High (display range: 0.5–99.5th percentiles; statistics use all samples).",
        "- `figure_3_4_bootstrap_ci.png`: mean differences and image-level 95% CIs of the primary comparisons.",
        "",
        "## 6. Summary",
        "",
        sentence_block(eff),
        "",
        "## 7. Limitations",
        "",
        "- CIs excluding zero do not explain detection-performance changes of FEPF or MDDM.",
        "- The higher DWT A4 values for smoke are not separated from overall brightness effects.",
        "- Dependence among adjacent video frames is not modeled because video IDs are unavailable; the CIs account only for image-level dependence.",
        "",
    ]
    (OUT / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global ROOT, SRC, OUT
    parser = argparse.ArgumentParser(description="Image-level cluster bootstrap CIs and Hedges' g for ROI features")
    parser.add_argument("--root", type=Path, default=ROOT, help="folder containing multifeature_train3000_235/")
    args = parser.parse_args()
    ROOT = args.root
    SRC = ROOT / "multifeature_train3000_235"
    OUT = ROOT / "roi_validation_32"
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_frames()
    audit_df = audit(df)
    desc = descriptive(df)
    eff = effects(df)
    write_tables(audit_df, desc, eff)
    plot_distributions(df)
    plot_ci(eff)
    write_report(df, audit_df, desc, eff)
    print(f"rows={len(df)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
