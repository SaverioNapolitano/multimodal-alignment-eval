"""
Wilcoxon signed-rank tests comparing per-folder correlations:
- CLIP vs LPIPS
- CLIP vs MS-SSIM

Input: correlation_report/correlations.csv produced by correlation_analysis.py
Output: correlation_report/wilcoxon_correlation_tests.csv
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
CORR_CSV = BASE_DIR / "correlation_report" / "correlations.csv"
OUTPUT_CSV = BASE_DIR / "correlation_report" / "wilcoxon_correlation_tests.csv"
OUTPUT_PLOT = BASE_DIR / "correlation_report" / "wilcoxon_correlation_tests.png"

METRIC_CLIP = "clip_cosine"
METRIC_LPIPS = "lpips"
METRIC_MSSSIM = "ms_ssim"
STATS = ["kendall_tau", "spearman", "pearson"]


def load_correlations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"folder", "metric"} | set(STATS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def paired_values(df: pd.DataFrame, stat: str, metric_a: str, metric_b: str) -> pd.DataFrame:
    """Return aligned per-folder values for the two metrics."""
    pivot = df.pivot(index="folder", columns="metric", values=stat)
    sub = pivot[[metric_a, metric_b]].dropna()
    return sub


def wilcoxon_for_stat(df: pd.DataFrame, stat: str, metric_a: str, metric_b: str) -> Dict:
    sub = paired_values(df, stat, metric_a, metric_b)
    n = len(sub)
    if n == 0:
        return {"stat": stat, "comparison": f"{metric_a}_vs_{metric_b}", "n": 0, "wilcoxon_stat": None, "pvalue": None}
    stat_val, pval = wilcoxon(sub[metric_a], sub[metric_b], zero_method="pratt", alternative="two-sided")
    return {
        "stat": stat,
        "comparison": f"{metric_a}_vs_{metric_b}",
        "n": n,
        "wilcoxon_stat": float(stat_val),
        "pvalue": float(pval),
    }


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    """Plot p-values for each comparison/stat."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x="stat", y="pvalue", hue="comparison")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="0.05")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wilcoxon p-value")
    ax.set_xlabel("Correlation statistic")
    ax.set_title("Wilcoxon signed-rank: per-image correlations")
    ax.legend(title="Comparison")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    df = load_correlations(CORR_CSV)
    results: List[Dict] = []
    for stat in STATS:
        results.append(wilcoxon_for_stat(df, stat, METRIC_CLIP, METRIC_LPIPS))
        results.append(wilcoxon_for_stat(df, stat, METRIC_CLIP, METRIC_MSSSIM))

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    plot_results(out_df, OUTPUT_PLOT)
    print(out_df)
    print(f"Wrote Wilcoxon results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
