"""
Compute agreement and correlation statistics between human rankings and automatic metrics.

Outputs (written next to this script by default):
- kendalls_w.csv : Kendall's W per image (inter-rater agreement across humans).
- correlations.csv : Kendall's Tau, Spearman, and Pearson correlations per folder per metric comparing
  human average ranks vs automatic metric ranks.
- spearman_bootstrap_ci.csv : bootstrap 95% CI for mean Spearman per metric.
- spearman_wilcoxon.csv : Wilcoxon signed-rank tests comparing Spearman values (CLIP vs others).
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon

BASE_DIR = Path(__file__).parent
HUMAN_RANKINGS = BASE_DIR / "rankings.txt"
AUTO_METRICS = BASE_DIR / "automatic_measures_results.csv"
OUTPUT_DIR = BASE_DIR / "correlation_report"

VARIANTS = ["a", "b", "c", "d"]
METRICS = [
    ("lpips", True),  # lower is better
    ("ms_ssim", False),
    ("clip_cosine", False),
]
METRIC_HATCH = {
    "lpips": "//",
    "ms_ssim": "xx",
    "clip_cosine": "..",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Kendall's W (human agreement) and correlations between human rankings and automatic metrics."
    )
    parser.add_argument("--rankings", type=Path, default=HUMAN_RANKINGS, help="Path to rankings.txt")
    parser.add_argument("--auto", type=Path, default=AUTO_METRICS, help="Path to automatic_measures_results.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Where to write CSV summaries and plots")
    return parser.parse_args()


# --- Human rankings parsing (mirrors human_preferences_analysis.py) ---
def load_blocks(path: Path, block_size: int = 15) -> List[List[Sequence[str]]]:
    blocks: List[List[Sequence[str]]] = []
    current: List[Sequence[str]] = []
    with path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue

            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed line {line_num}: {line}")
            ranking = parts[1:]
            if sorted(ranking) != sorted(VARIANTS):
                raise ValueError(f"Invalid ranking on line {line_num}: {ranking}")

            current.append(ranking)
            if len(current) == block_size:
                blocks.append(current)
                current = []

    if current:
        blocks.append(current)

    # Validate block sizes
    for i, blk in enumerate(blocks, start=1):
        if len(blk) != block_size:
            raise ValueError(f"Block {i} has {len(blk)} lines (expected {block_size}).")
    return blocks


def human_rank_matrix(blocks: List[List[Sequence[str]]], folder_idx: int) -> np.ndarray:
    """
    Return matrix shape (num_raters, num_variants) of ranks (1=best) for the given folder index (0-based).
    Columns follow VARIANTS order.
    """
    num_raters = len(blocks)
    mat = np.zeros((num_raters, len(VARIANTS)), dtype=int)
    for rater_idx, block in enumerate(blocks):
        ranking = block[folder_idx]
        for pos, variant in enumerate(ranking, start=1):
            mat[rater_idx, VARIANTS.index(variant)] = pos
    return mat


def kendalls_w(rank_mat: np.ndarray) -> float:
    """Compute Kendall's W for a rank matrix (raters x items)."""
    m, n = rank_mat.shape  # m raters, n items
    # Sum ranks per item
    R = rank_mat.sum(axis=0)
    R_bar = R.mean()
    S = np.sum((R - R_bar) ** 2)
    denom = m ** 2 * (n ** 3 - n)
    return float(12 * S / denom) if denom else np.nan


# --- Automatic metrics helpers ---
def load_auto(auto_path: Path) -> pd.DataFrame:
    df = pd.read_csv(auto_path)
    expected = {"folder", "variant"} | {m for m, _ in METRICS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {auto_path}: {sorted(missing)}")
    return df


def metric_ranks_for_folder(df: pd.DataFrame, folder: str) -> Dict[str, pd.Series]:
    """Return per-metric rank Series indexed by variant for the given folder."""
    folder_df = df[df["folder"].astype(str) == str(folder)].set_index("variant")
    ranks = {}
    for metric, ascending in METRICS:
        ranks[metric] = folder_df[metric].rank(ascending=ascending, method="first")
    return ranks


# --- Correlation computation ---
def correlations_for_folder(
    folder_idx: int,
    human_blocks: List[List[Sequence[str]]],
    auto_df: pd.DataFrame,
) -> List[Dict]:
    folder_name = str(folder_idx + 1).zfill(2)
    # Human mean ranks (1=best)
    hrank_mat = human_rank_matrix(human_blocks, folder_idx)
    human_mean_rank = hrank_mat.mean(axis=0)
    human_rank_series = pd.Series(human_mean_rank, index=VARIANTS, name="human_mean_rank")

    # Auto ranks
    auto_ranks = metric_ranks_for_folder(auto_df, folder_idx + 1)

    rows = []
    for metric, _ in METRICS:
        auto_rank = auto_ranks[metric]
        # Align by variant
        h = human_rank_series
        a = auto_rank.reindex(VARIANTS)
        ktau, _ = kendalltau(h, a)
        spearman, _ = spearmanr(h, a)
        pearson, _ = pearsonr(h, a)
        rows.append(
            {
                "folder": folder_name,
                "metric": metric,
                "kendall_tau": ktau,
                "spearman": spearman,
                "pearson": pearson,
            }
        )
    return rows


def plot_kendalls_w(df: pd.DataFrame, output_path: Path):
    """Bar plot of Kendall's W per image."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="folder", y="kendalls_w", ax=ax, color="tab:blue")
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color="gray", linestyle="--", linewidth=1, label="0.7 (strong)")
    ax.set_ylabel("Kendall's W (agreement)")
    ax.set_xlabel("Image")
    ax.set_title("Inter-rater agreement per image (Kendall's W)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlations(corr_df: pd.DataFrame, output_dir: Path):
    """Create plots for correlations: per-folder grouped bars (one plot per stat) and metric means."""
    long_df = corr_df.melt(id_vars=["folder", "metric"], var_name="stat", value_name="value")

    # Per-folder plots, split by correlation stat for clarity
    for stat in ["kendall_tau", "spearman", "pearson"]:
        subset = long_df[long_df["stat"] == stat]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=subset, x="folder", y="value", hue="metric", ax=ax)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"{stat.replace('_', ' ').title()} correlation")
        ax.set_title(f"{stat.replace('_', ' ').title()} vs human mean ranks (per folder)")
        fig.tight_layout()
        fig.savefig(output_dir / f"correlations_per_folder_{stat}.png", dpi=300)
        plt.close(fig)

    # Metric-level averages
    avg = long_df.groupby(["metric", "stat"])["value"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=avg, x="metric", y="value", hue="stat", ax=ax)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Average correlation across folders")
    ax.set_title("Average correlations (human vs automatic metrics)")
    fig.tight_layout()
    fig.savefig(output_dir / "correlations_average.png", dpi=300)
    plt.close(fig)

    # Save averages to CSV
    avg.pivot(index="metric", columns="stat", values="value").reset_index().to_csv(
        output_dir / "correlations_average.csv", index=False
    )

    # Combined per-folder summary (mean ± min/max across all three stats), candle-style error bars
    plot_correlations_combined(corr_df, output_dir / "correlations_per_folder.png")


def plot_correlations_combined(corr_df: pd.DataFrame, output_path: Path):
    """Per-image correlation summary: mean with min/max whiskers across Tau/Spearman/Pearson."""
    stats = ["kendall_tau", "spearman", "pearson"]
    metric_order = [m for m, _ in METRICS]

    melted = corr_df.melt(id_vars=["folder", "metric"], var_name="stat", value_name="value")
    melted = melted[melted["stat"].isin(stats)]

    summary = (
        melted.groupby(["folder", "metric"])["value"]
        .agg(mean="mean", min="min", max="max")
        .reset_index()
    )

    folders = sorted(summary["folder"].unique(), key=lambda x: int(x))
    x_positions = np.arange(len(folders))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, metric in enumerate(metric_order):
        sub = summary[summary["metric"] == metric]
        # Align by folder
        sub = sub.set_index("folder").reindex(folders)
        mean = sub["mean"].to_numpy()
        ymin = mean - sub["min"].to_numpy()
        ymax = sub["max"].to_numpy() - mean
        offset = (idx - (len(metric_order) - 1) / 2) * width
        ax.bar(
            x_positions + offset,
            mean,
            width=width,
            yerr=[ymin, ymax],
            capsize=4,
            color="white",
            edgecolor="black",
            hatch=METRIC_HATCH.get(metric, ""),
            label=metric,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(folders, rotation=45, ha="right")
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Correlation (mean ± range across Tau/Spearman/Pearson)")
    ax.set_xlabel("Image")
    ax.set_title("Per-image correlation summary")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap mean confidence interval."""
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def spearman_bootstrap_ci(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute bootstrap CI for mean Spearman per metric."""
    rows = []
    for metric, _ in METRICS:
        vals = corr_df[corr_df["metric"] == metric]["spearman"].dropna().to_numpy()
        mean = float(vals.mean()) if len(vals) else np.nan
        lower, upper = bootstrap_mean_ci(vals) if len(vals) else (np.nan, np.nan)
        rows.append({"metric": metric, "mean_spearman": mean, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)


def wilcoxon_spearman(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank (one-sided, greater) for Spearman: CLIP vs LPIPS and CLIP vs MS-SSIM."""
    rows = []
    pivot = corr_df.pivot(index="folder", columns="metric", values="spearman")
    comparisons = [("clip_cosine", "lpips"), ("clip_cosine", "ms_ssim")]
    for a, b in comparisons:
        paired = pivot[[a, b]].dropna()
        n = len(paired)
        if n == 0:
            stat = pval = np.nan
        else:
            stat, pval = wilcoxon(paired[a], paired[b], alternative="greater", zero_method="pratt")
        rows.append({"comparison": f"{a}_gt_{b}", "n": n, "wilcoxon_stat": stat, "pvalue": pval})
    return pd.DataFrame(rows)


def plot_spearman_ci(boot_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of mean Spearman with bootstrap 95% CI."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    errors = [
        boot_df["mean_spearman"] - boot_df["ci_lower"],
        boot_df["ci_upper"] - boot_df["mean_spearman"],
    ]
    ax.bar(
        boot_df["metric"],
        boot_df["mean_spearman"],
        yerr=errors,
        capsize=4,
        color="white",
        edgecolor="black",
        hatch=["..", "//", "xx"],
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Mean Spearman (human ranks vs metric ranks)")
    ax.set_xlabel("Metric")
    ax.set_title("Mean Spearman with 95% bootstrap CI")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(args.rankings)
    auto_df = load_auto(args.auto)

    # Kendall's W per folder
    kendall_w_rows = []
    num_folders = len(blocks[0])
    for folder_idx in range(num_folders):
        rank_mat = human_rank_matrix(blocks, folder_idx)
        kendall_w_rows.append(
            {
        "folder": str(folder_idx + 1).zfill(2),
                "kendalls_w": kendalls_w(rank_mat),
            }
        )
    kendall_w_df = pd.DataFrame(kendall_w_rows)
    kendall_w_df.to_csv(output_dir / "kendalls_w.csv", index=False)
    plot_kendalls_w(kendall_w_df, output_dir / "kendalls_w.png")

    # Correlations per folder per metric
    corr_rows: List[Dict] = []
    for folder_idx in range(num_folders):
        corr_rows.extend(correlations_for_folder(folder_idx, blocks, auto_df))
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(output_dir / "correlations.csv", index=False)
    plot_correlations(corr_df, output_dir)

    # Spearman bootstrap CI and Wilcoxon tests (CLIP vs others)
    boot_df = spearman_bootstrap_ci(corr_df)
    boot_df.to_csv(output_dir / "spearman_bootstrap_ci.csv", index=False)
    wilcoxon_df = wilcoxon_spearman(corr_df)
    wilcoxon_df.to_csv(output_dir / "spearman_wilcoxon.csv", index=False)
    plot_spearman_ci(boot_df, output_dir / "spearman_mean_ci.png")

    print(f"Wrote Kendall's W to {output_dir / 'kendalls_w.csv'}")
    print(f"Wrote correlations to {output_dir / 'correlations.csv'}")
    print(f"Wrote Spearman bootstrap CI to {output_dir / 'spearman_bootstrap_ci.csv'}")
    print(f"Wrote Wilcoxon Spearman tests to {output_dir / 'spearman_wilcoxon.csv'}")


if __name__ == "__main__":
    main()
