"""
Summarize the automatic image metrics and generate human-friendly plots.

Outputs:
- automatic_measures_report/metric_means.png
- automatic_measures_report/metric_winners_heatmap.png
- automatic_measures_report/human_vs_automatic.png (if rankings.txt is found)
- CSV summaries alongside the plots for downstream analysis.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "automatic_measures_results.csv"
DEFAULT_RANKINGS = BASE_DIR / "rankings.txt"
OUTPUT_DIR = BASE_DIR / "automatic_measures_report"

VARIANT_MAP = {"a": "gg", "b": "cg", "c": "gc", "d": "cc"}
VARIANT_KEYS = list(VARIANT_MAP.keys())
VARIANTS = [VARIANT_MAP[k] for k in VARIANT_KEYS]
METRICS = [
    ("lpips", True),  # lower is better
    ("ms_ssim", False),
    ("clip_cosine", False),
]
SOURCES = [
    "human_top1_percent",
    "clip_global_win_percent",
    "lpips_global_win_percent",
    "ms_ssim_global_win_percent",
]
SOURCE_LABEL = {
    "human_top1_percent": "Human top-1 %",
    "clip_global_win_percent": "CLIP global win %",
    "lpips_global_win_percent": "LPIPS global win %",
    "ms_ssim_global_win_percent": "MS-SSIM global win %",
}
SOURCE_SHORT = {
    "human_top1_percent": "Human",
    "clip_global_win_percent": "CLIP",
    "lpips_global_win_percent": "LPIPS",
    "ms_ssim_global_win_percent": "MS-SSIM",
}
HATCHES_VARIANT = {"gg": "//", "cg": "\\\\", "gc": "xx", "cc": ".."}
HATCHES_SOURCE = {
    "human_top1_percent": "//",
    "clip_global_win_percent": "\\\\",
    "lpips_global_win_percent": "xx",
    "ms_ssim_global_win_percent": "..",
}
MARKERS_VARIANT = {"gg": "o", "cg": "s", "gc": "^", "cc": "D"}


def normalize_variant(variant: str) -> str:
    return VARIANT_MAP.get(variant, variant)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make plots from automatic_measures_results.csv and prep comparison data for human preferences."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to automatic_measures_results.csv (default: alongside this script).",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=DEFAULT_RANKINGS,
        help="Path to rankings.txt used by human_preferences_analysis.py (default: ../rankings.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write plots and summaries (default: automatic_measures_report).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots interactively after saving them.",
    )
    return parser.parse_args()


def load_automatic_measures(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"folder", "variant", "lpips", "ms_ssim", "clip_cosine"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    raw_variants = set(df["variant"])
    if raw_variants & set(VARIANT_KEYS):
        warnings.warn(
            "Legacy variants (a-d) detected in automatic measures; mapping to gg/cg/gc/cc.",
            stacklevel=2,
        )
    df["variant"] = df["variant"].map(normalize_variant)
    unknown = set(df["variant"]) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unexpected variants in {csv_path}: {sorted(unknown)}")
    return df


def summarize_automatic(df: pd.DataFrame):
    """Return (variant_means, best_counts) data frames.

    best_counts now reports a global percentage of wins per variant+metric,
    i.e., wins divided by the total number of rows for that metric (all folders).
    """
    variant_means = (
        df.groupby("variant")[["lpips", "ms_ssim", "clip_cosine"]]
        .mean()
        .reindex(VARIANTS)
        .reset_index()
    )

    best_rows = []
    total_rows = len(df)
    for metric, ascending in METRICS:
        idx = df.groupby("folder")[metric].idxmin() if ascending else df.groupby("folder")[metric].idxmax()
        winners = df.loc[idx, "variant"].value_counts().reindex(VARIANTS, fill_value=0)
        for variant in VARIANTS:
            count = int(winners[variant])
            best_rows.append(
                {
                    "metric": metric,
                    "variant": variant,
                    "best_count": count,
                    "global_best_percent": (count / total_rows * 100.0) if total_rows else 0.0,
                }
            )

    best_counts = pd.DataFrame(best_rows)
    return variant_means, best_counts


def plot_metric_means(variant_means: pd.DataFrame, output_path: Path):
    long_df = variant_means.melt(id_vars="variant", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=long_df,
        x="metric",
        y="score",
        hue="variant",
        hue_order=VARIANTS,
        palette="Greys",
        ax=ax,
    )
    # Apply hatches/edges for B&W readability
    keys = [v for _ in long_df["metric"].unique() for v in VARIANTS]
    for patch, key in zip(ax.patches, keys):
        patch.set_hatch(HATCHES_VARIANT.get(key, ""))
        patch.set_edgecolor("black")
        patch.set_facecolor("white")
    ax.set_title("Average automatic scores (LPIPS lower is better)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=HATCHES_VARIANT.get(v, ""),
            label=v,
        )
        for v in VARIANTS
    ]
    ax.legend(handles=handles, title="Variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_best_heatmap(best_counts: pd.DataFrame, output_path: Path):
    heat = best_counts.pivot(index="variant", columns="metric", values="global_best_percent").loc[VARIANTS]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="Greys",
        cbar_kws={"label": "Global win % (wins / all rows)"},
        ax=ax,
    )
    ax.set_title("Global win percentage per metric")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# Human preference helpers (adapted from human_preferences_analysis.py without plotting)
def load_human_blocks(path: Path, block_size: int = 15) -> List[List[Sequence[str]]]:
    blocks: List[List[Sequence[str]]] = []
    current: List[Sequence[str]] = []
    warned_legacy = False
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
            if sorted(ranking) != sorted(VARIANT_KEYS):
                raise ValueError(f"Invalid ranking on line {line_num}: {ranking}")
            if not warned_legacy:
                warnings.warn(
                    "Legacy rankings (a-d) detected; mapping to gg/cg/gc/cc.",
                    stacklevel=2,
                )
                warned_legacy = True
            current.append([VARIANT_MAP[v] for v in ranking])
            if len(current) == block_size:
                blocks.append(current)
                current = []

    if current:
        blocks.append(current)

    return blocks


def human_top1_stats(blocks: Iterable[Iterable[Sequence[str]]]) -> pd.DataFrame:
    """Return counts and percentages for how often each variant is ranked #1 by humans."""
    counts: Dict[str, int] = {v: 0 for v in VARIANTS}
    total = 0
    for block in blocks:
        for ranking in block:
            top = ranking[0]
            counts[top] += 1
            total += 1
    if total == 0:
        raise ValueError("No human rankings found; cannot compute top-1 preferences.")
    data = [
        {
            "variant": variant,
            "human_top1_percent": counts[variant] / total * 100.0,
        }
        for variant in VARIANTS
    ]
    return pd.DataFrame(data)


def build_comparison(best_counts: pd.DataFrame, human_top1: pd.DataFrame) -> pd.DataFrame:
    """Combine automatic winners and human top-1 preference into one table."""
    auto = best_counts.pivot(index="variant", columns="metric", values="global_best_percent").reindex(VARIANTS)
    auto = auto.rename(
        columns={
            "lpips": "lpips_global_win_percent",
            "ms_ssim": "ms_ssim_global_win_percent",
            "clip_cosine": "clip_global_win_percent",
        }
    )
    combined = auto.join(human_top1.set_index("variant"))
    # Add deltas (automatic % minus human %)
    combined["lpips_minus_human"] = combined["lpips_global_win_percent"] - combined["human_top1_percent"]
    combined["ms_ssim_minus_human"] = combined["ms_ssim_global_win_percent"] - combined["human_top1_percent"]
    combined["clip_minus_human"] = combined["clip_global_win_percent"] - combined["human_top1_percent"]
    combined = combined.reset_index()
    return combined


def plot_human_vs_auto(comparison: pd.DataFrame, output_path: Path):
    """Overall comparison: average percent per source (no per-variant breakdown)."""
    tidy = comparison[SOURCES + ["variant"]].melt(id_vars="variant", var_name="source", value_name="percent")
    overall = tidy.groupby("source")["percent"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=overall,
        x="source",
        y="percent",
        order=SOURCES,
        palette="Greys",
        ax=ax,
    )
    for patch, key in zip(ax.patches, SOURCES):
        hatch = HATCHES_SOURCE.get(key, "")
        patch.set_hatch(hatch)
        patch.set_edgecolor("black")
        patch.set_facecolor("white")
    ax.set_ylabel("Percent")
    ax.set_xlabel("Source")
    ax.set_title("Overall human vs automatic win percentages (mean across variants)")
    handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=HATCHES_SOURCE.get(s, ""),
            label=SOURCE_LABEL.get(s, s),
        )
        for s in SOURCES
    ]
    ax.set_xticklabels([SOURCE_LABEL.get(s, s) for s in SOURCES], rotation=20, ha="right")
    ax.legend(handles=handles, title="Source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_variant_bars(comparison: pd.DataFrame, output_path: Path):
    """Grouped bars per variant: human top1 % and automatic global win % per metric."""
    tidy = comparison[
        [
            "variant",
            "human_top1_percent",
            "clip_global_win_percent",
            "lpips_global_win_percent",
            "ms_ssim_global_win_percent",
        ]
    ].melt(id_vars="variant", var_name="source", value_name="percent")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=tidy,
        x="variant",
        y="percent",
        hue="source",
        hue_order=SOURCES,
        palette="Greys",
        ax=ax,
    )
    # Apply styling/labels per source container to avoid misalignment
    for source, container in zip(SOURCES, ax.containers):
        for patch in container:
            patch.set_hatch(HATCHES_SOURCE.get(source, ""))
            patch.set_edgecolor("black")
            patch.set_facecolor("white")
    ax.set_ylabel("Percent")
    ax.set_title("Human vs. automatic win percentages per variant")
    handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=HATCHES_SOURCE.get(s, ""),
            label=SOURCE_LABEL.get(s, s),
        )
        for s in SOURCES
    ]
    ax.legend(handles=handles, title="Source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_human_vs_auto_scatter(comparison: pd.DataFrame, output_path: Path):
    """Scatter plots: human_top1_percent vs each automatic metric global win percent."""
    metric_cols = [
        ("lpips_global_win_percent", "LPIPS (global win %)"),
        ("ms_ssim_global_win_percent", "MS-SSIM (global win %)"),
        ("clip_global_win_percent", "CLIP cosine (global win %)"),
    ]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(12, 4), sharey=True)
    human = comparison["human_top1_percent"]
    for ax, (col, title) in zip(axes, metric_cols):
        for _, row in comparison.iterrows():
            marker = MARKERS_VARIANT.get(row["variant"], "o")
            ax.scatter(
                row["human_top1_percent"],
                row[col],
                marker=marker,
                color="white",
                edgecolor="black",
                s=80,
                label=row["variant"],
            )
        lims = [
            min(human.min(), comparison[col].min()) - 2,
            max(human.max(), comparison[col].max()) + 2,
        ]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Human top-1 percent")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend entries
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Variant", frameon=False)
    axes[0].set_ylabel("Automatic metric global win percent")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def rank_distributions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    For each metric, compute distribution of ranks per variant.
    Returns mapping metric -> DataFrame (rows=variant, cols=rank 1-4, values=global percent of rows).
    """
    results: Dict[str, pd.DataFrame] = {}
    total_rows = len(df)  # folders * variants
    for metric, ascending in METRICS:
        metric_df = df[["folder", "variant", metric]].copy()
        metric_df["rank"] = metric_df.groupby("folder")[metric].rank(
            ascending=ascending, method="first"
        )
        counts = (
            metric_df.groupby(["variant", "rank"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=VARIANTS, columns=[1.0, 2.0, 3.0, 4.0], fill_value=0)
        )
        perc = counts / total_rows * 100.0 if total_rows else counts
        perc.columns = [str(int(c)) for c in perc.columns]
        perc = perc.reset_index()
        results[metric] = perc
    return results


def plot_rank_heatmap(dist_df: pd.DataFrame, metric: str, output_path: Path):
    """Heatmap of rank distribution per variant for one metric."""
    heat = dist_df.set_index("variant")[["1", "2", "3", "4"]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".1f",
        cmap="Greys",
        cbar_kws={"label": "Global % of rows"},
        ax=ax,
    )
    ax.set_title(f"{metric} rank distribution (1=best)")
    ax.set_xlabel("Rank position")
    ax.set_ylabel("Variant")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")

    df = load_automatic_measures(args.csv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_means, best_counts = summarize_automatic(df)
    variant_means.to_csv(output_dir / "variant_means.csv", index=False)
    best_counts.to_csv(output_dir / "best_by_metric.csv", index=False)

    plot_metric_means(variant_means, output_dir / "metric_means.png")
    plot_best_heatmap(best_counts, output_dir / "metric_winners_heatmap.png")

    comparison = None
    if args.rankings and Path(args.rankings).exists():
        blocks = load_human_blocks(Path(args.rankings))
        human_top1 = human_top1_stats(blocks)
        human_top1.to_csv(output_dir / "human_top1_percent.csv", index=False)
        comparison = build_comparison(best_counts, human_top1)
        comparison.to_csv(output_dir / "human_vs_automatic.csv", index=False)
        plot_variant_bars(comparison, output_dir / "human_vs_automatic_variants.png")
        plot_human_vs_auto(comparison, output_dir / "human_vs_automatic.png")
        plot_human_vs_auto_scatter(comparison, output_dir / "human_vs_automatic_scatter.png")

        # Rank distributions per metric
        rank_dist = rank_distributions(df)
        for metric, dist_df in rank_dist.items():
            dist_df.to_csv(output_dir / f"rank_distribution_{metric}.csv", index=False)
            plot_rank_heatmap(dist_df, metric, output_dir / f"rank_distribution_{metric}.png")
    else:
        print(f"Human rankings file not found at {args.rankings}; skipping human comparison.")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
