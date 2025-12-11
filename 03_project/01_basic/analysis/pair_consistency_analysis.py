"""
Analyze ordering consistency for variant pairs (a vs b and c vs d) across raters and reference images.

Outputs:
- pair_consistency_report/pair_differences.csv : per-rater rank differences for each pair.
- pair_consistency_report/folder_summary.csv : per-folder aggregation (order flip rate, mean differences).
- pair_consistency_report/pair_summary.csv : dataset-level summary per pair.
- Plots saved to pair_consistency_report/*.png visualizing flip rates and rank differences.

Definitions:
- rank_diff = rank(first_variant) - rank(second_variant); negative => first variant ranked better.
- flip_rate = fraction of ratings not matching the majority ordering within a folder (0 = fully consistent).
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).parent
DEFAULT_RANKINGS = BASE_DIR / "rankings.txt"
OUTPUT_DIR = BASE_DIR / "pair_consistency_report"
VARIANTS = ["a", "b", "c", "d"]
PAIRS: List[Tuple[str, str]] = [("a", "b"), ("c", "d")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ordering consistency between variant pairs across raters and reference images."
    )
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS, help="Path to rankings.txt")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory where CSVs and plots will be written"
    )
    return parser.parse_args()


def load_blocks(path: Path, block_size: int = 15) -> List[List[Sequence[str]]]:
    """Load rankings into blocks of `block_size` lines (one evaluator per block)."""
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

    for i, blk in enumerate(blocks, start=1):
        if len(blk) != block_size:
            raise ValueError(f"Block {i} has {len(blk)} lines (expected {block_size}).")
    return blocks


def blocks_to_rank_df(blocks: List[List[Sequence[str]]]) -> pd.DataFrame:
    """Convert ranking blocks to a long DataFrame with one row per rater/folder/variant."""
    records: List[Dict] = []
    for rater_idx, block in enumerate(blocks):
        for folder_idx, ranking in enumerate(block):
            folder_name = str(folder_idx + 1).zfill(2)
            for pos, variant in enumerate(ranking, start=1):
                records.append(
                    {
                        "rater": rater_idx,
                        "folder": folder_name,
                        "variant": variant,
                        "rank": pos,
                    }
                )
    return pd.DataFrame.from_records(records)


def compute_pair_differences(rank_df: pd.DataFrame, pairs: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    """Return DataFrame with rank differences for each pair per rater and folder."""
    diffs: List[Dict] = []
    pivot = rank_df.pivot_table(index=["rater", "folder"], columns="variant", values="rank")
    for first, second in pairs:
        if first not in pivot.columns or second not in pivot.columns:
            raise ValueError(f"Missing variants in rank data: {first}, {second}")
        sub = pivot[[first, second]].dropna()
        for (rater, folder), row in sub.iterrows():
            diff = row[first] - row[second]  # negative => first variant ranked better
            diffs.append(
                {
                    "rater": rater,
                    "folder": folder,
                    "pair": f"{first}_vs_{second}",
                    "first": first,
                    "second": second,
                    "rank_diff": diff,
                    "order": "first_better" if diff < 0 else ("second_better" if diff > 0 else "tie"),
                }
            )
    return pd.DataFrame.from_records(diffs)


def summarize_by_folder(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pair differences per folder."""
    rows: List[Dict] = []
    for (pair, folder), group in pair_df.groupby(["pair", "folder"]):
        n = len(group)
        first_better = (group["rank_diff"] < 0).sum()
        second_better = (group["rank_diff"] > 0).sum()
        ties = (group["rank_diff"] == 0).sum()
        majority = max(first_better, second_better, ties) if n else 0
        first_variant = group["first"].iloc[0]
        second_variant = group["second"].iloc[0]
        if n == 0:
            majority_order = "n/a"
        elif ties == majority:
            majority_order = "tie"
        elif first_better == majority:
            majority_order = f"{first_variant} > {second_variant}"
        else:
            majority_order = f"{second_variant} > {first_variant}"
        rows.append(
            {
                "pair": pair,
                "folder": folder,
                "first": first_variant,
                "second": second_variant,
                "n_raters": n,
                "first_better": first_better,
                "second_better": second_better,
                "ties": ties,
                "first_share": first_better / n if n else np.nan,
                "second_share": second_better / n if n else np.nan,
                "flip_rate": 1 - (majority / n) if n else np.nan,
                "mean_rank_diff": group["rank_diff"].mean(),
                "mean_abs_rank_diff": group["rank_diff"].abs().mean(),
                "std_rank_diff": group["rank_diff"].std(ddof=0),
                "majority_order": majority_order,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_pair(folder_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate folder-level stats into a dataset-level summary per pair."""
    rows: List[Dict] = []
    for pair, group in folder_summary.groupby("pair"):
        # Unweighted averages across folders
        rows.append(
            {
                "pair": pair,
                "mean_flip_rate": group["flip_rate"].mean(),
                "mean_abs_rank_diff": group["mean_abs_rank_diff"].mean(),
                "mean_rank_diff": group["mean_rank_diff"].mean(),
                "mean_first_share": group["first_share"].mean(),
                "mean_second_share": group["second_share"].mean(),
                "weighted_flip_rate": np.average(group["flip_rate"], weights=group["n_raters"]),
                "weighted_mean_rank_diff": np.average(group["mean_rank_diff"], weights=group["n_raters"]),
                "weighted_mean_abs_rank_diff": np.average(group["mean_abs_rank_diff"], weights=group["n_raters"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_global(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across all folders/raters for each pair.

    flip_rate here is computed globally: 1 - (majority order count / total observations).
    """
    rows: List[Dict] = []
    for pair, group in pair_df.groupby("pair"):
        n = len(group)
        first_better = (group["rank_diff"] < 0).sum()
        second_better = (group["rank_diff"] > 0).sum()
        ties = (group["rank_diff"] == 0).sum()
        majority = max(first_better, second_better, ties) if n else 0
        rows.append(
            {
                "pair": pair,
                "total_votes": n,
                "first_better": first_better,
                "second_better": second_better,
                "ties": ties,
                "global_flip_rate": 1 - (majority / n) if n else np.nan,
                "global_mean_rank_diff": group["rank_diff"].mean(),
                "global_mean_abs_rank_diff": group["rank_diff"].abs().mean(),
                "global_std_rank_diff": group["rank_diff"].std(ddof=0),
            }
        )
    return pd.DataFrame(rows)


def compute_preference_table(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a single-row table with percentages for each ordering within its own pair:
    - a > b
    - b > a
    - c > d
    - d > c
    Percentages are computed independently per pair (ties reduce the total per-pair sum).
    """
    def pair_pct(pair_name: str, condition: pd.Series) -> float:
        subset = pair_df[pair_df["pair"] == pair_name]
        total = len(subset)
        if total == 0:
            return np.nan
        return 100.0 * condition.loc[subset.index].sum() / total

    is_first_better = pair_df["rank_diff"] < 0
    is_second_better = pair_df["rank_diff"] > 0

    row = {
        "a>b": pair_pct("a_vs_b", is_first_better),
        "b>a": pair_pct("a_vs_b", is_second_better),
        "c>d": pair_pct("c_vs_d", is_first_better),
        "d>c": pair_pct("c_vs_d", is_second_better),
    }
    return pd.DataFrame([row])


def plot_flip_rates(folder_summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of flip rates (order instability) per folder."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 4))
    order = sorted(folder_summary["folder"].unique(), key=lambda x: int(x))
    hue_order = ["a_vs_b", "c_vs_d"]
    ax = sns.barplot(
        data=folder_summary,
        x="folder",
        y="flip_rate",
        hue="pair",
        order=order,
        hue_order=hue_order,
        palette={"a_vs_b": "tab:blue", "c_vs_d": "tab:orange"},
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Flip rate (share not in majority order)")
    ax.set_xlabel("Reference image folder")
    ax.set_title("Order stability per image")
    ax.legend(title="Pair")

    # Annotate bars with the majority order per folder/pair (e.g., "a > b", "tie")
    label_map = {(row["pair"], row["folder"]): row["majority_order"] for _, row in folder_summary.iterrows()}
    for container, pair in zip(ax.containers, hue_order):
        for bar, folder in zip(container, order):
            label = label_map.get((pair, folder))
            if not label:
                continue
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.text(x, y + 0.02, label, ha="center", va="bottom", fontsize=8, rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "flip_rate_per_folder.png", dpi=300)
    plt.close()


def plot_rank_diff_distribution(pair_df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of rank differences for each pair across all raters and folders."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(data=pair_df, x="pair", y="rank_diff", palette="Set2")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Rank difference (first - second)")
    ax.set_title("Rank difference distribution by pair")
    plt.tight_layout()
    plt.savefig(output_dir / "rank_difference_boxplot.png", dpi=300)
    plt.close()


def plot_mean_rank_diff_heatmap(folder_summary: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of mean rank differences per folder (direction and magnitude), B/W friendly."""
    pivot = folder_summary.pivot(index="pair", columns="folder", values="mean_rank_diff")
    plt.figure(figsize=(12, 3))
    # Use a grayscale map centered at 0 to stay printer-friendly
    cmap = sns.color_palette("Greys", as_cmap=True)
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean rank difference (first - second)"},
        linewidths=0.5,
        linecolor="black",
    )
    ax.set_xlabel("Reference image folder")
    ax.set_ylabel("Pair")
    ax.set_title("Mean rank difference by folder")
    # Adjust annotation color based on background luminance for readability in grayscale
    norm = ax.collections[0].norm
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except ValueError:
            continue
        rgba = cmap(norm(val))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text.set_color("black" if luminance > 0.5 else "white")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_rank_difference_heatmap.png", dpi=300)
    plt.close()


def plot_pair_summary(pair_summary: pd.DataFrame, global_summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot dataset-level summaries (flip rate and rank difference) per pair."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Flip rates
    sns.barplot(
        data=pair_summary,
        x="pair",
        y="weighted_flip_rate",
        ax=axes[0],
        palette={"a_vs_b": "tab:blue", "c_vs_d": "tab:orange"},
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Weighted flip rate across folders")
    axes[0].set_ylabel("Flip rate (weighted by raters)")
    axes[0].set_xlabel("Pair")

    # Absolute rank differences
    sns.barplot(
        data=pair_summary,
        x="pair",
        y="weighted_mean_abs_rank_diff",
        ax=axes[1],
        palette={"a_vs_b": "tab:blue", "c_vs_d": "tab:orange"},
    )
    axes[1].set_title("Weighted mean |rank diff| across folders")
    axes[1].set_ylabel("Mean absolute rank difference")
    axes[1].set_xlabel("Pair")

    plt.tight_layout()
    fig.savefig(output_dir / "pair_summary.png", dpi=300)
    plt.close(fig)

    # Global summary table snapshot (saved as CSV already); add a simple bar for global flip rate.
    plt.figure(figsize=(4, 4))
    sns.barplot(
        data=global_summary,
        x="pair",
        y="global_flip_rate",
        palette={"a_vs_b": "tab:blue", "c_vs_d": "tab:orange"},
    )
    plt.ylim(0, 1)
    plt.ylabel("Global flip rate")
    plt.xlabel("Pair")
    plt.title("Global flip rate (all folders/raters)")
    plt.tight_layout()
    plt.savefig(output_dir / "global_flip_rate.png", dpi=300)
    plt.close()


def plot_preference_table(pref_table: pd.DataFrame, output_dir: Path) -> None:
    """Plot the single-row preference percentages as a bar chart."""
    sns.set_theme(style="whitegrid")
    order = ["a>b", "b>a", "c>d", "d>c"]
    melted = pref_table.melt(value_vars=order, var_name="ordering", value_name="percentage")
    palette = {
        "a>b": "tab:blue",
        "b>a": "tab:cyan",
        "c>d": "tab:orange",
        "d>c": "gold",
    }
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=melted, x="ordering", y="percentage", palette=palette, order=order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage within pair (%)")
    ax.set_xlabel("Ordering (lower rank = better)")
    ax.set_title("Ordering percentages (per pair)")
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "preference_table.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = load_blocks(args.rankings)
    rank_df = blocks_to_rank_df(blocks)
    pair_df = compute_pair_differences(rank_df, PAIRS)
    folder_summary = summarize_by_folder(pair_df)
    pair_summary = summarize_by_pair(folder_summary)
    global_summary = summarize_global(pair_df)
    preference_table = compute_preference_table(pair_df)

    pair_df.to_csv(output_dir / "pair_differences.csv", index=False)
    folder_summary.sort_values(["pair", "folder"]).to_csv(output_dir / "folder_summary.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    global_summary.to_csv(output_dir / "global_summary.csv", index=False)
    preference_table.to_csv(output_dir / "preference_table.csv", index=False)

    plot_flip_rates(folder_summary, output_dir)
    plot_rank_diff_distribution(pair_df, output_dir)
    plot_mean_rank_diff_heatmap(folder_summary, output_dir)
    plot_pair_summary(pair_summary, global_summary, output_dir)
    plot_preference_table(preference_table, output_dir)

    print(f"Wrote reports to {output_dir}")


if __name__ == "__main__":
    main()
