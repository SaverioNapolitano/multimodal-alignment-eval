import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import frontiers_style
frontiers_style.apply()
from pathlib import Path
import warnings

# File lives alongside this script
FILE_PATH = Path(__file__).parent / "rankings.txt"
OUTPUT_DIR = Path(__file__).parent / "human_preferences_report"
VARIANT_MAP = {"a": "gg", "b": "cg", "c": "gc", "d": "cc"}
VARIANT_KEYS = list(VARIANT_MAP.keys())
VARIANTS = [VARIANT_MAP[k] for k in VARIANT_KEYS]
positions = ['1', '2', '3', '4']


def load_blocks(path: Path, block_size: int = 15):
    """Load rankings into blocks of `block_size` lines (one evaluator per block)."""
    blocks, current = [], []
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
            if sorted(ranking) != VARIANT_KEYS:
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

    if not blocks:
        raise ValueError("No rankings found in file.")

    for i, blk in enumerate(blocks, start=1):
        if len(blk) != block_size:
            raise ValueError(f"Block {i} has {len(blk)} lines (expected {block_size}).")

    return blocks


def tally_blocks(blocks):
    """Return count matrix for given blocks: rows=variant, cols=position."""
    counts = np.zeros((4, 4), dtype=int)
    for block in blocks:
        for ranking in block:
            for pos, letter in enumerate(ranking):
                counts[VARIANTS.index(letter), pos] += 1
    return counts


def to_percentages(counts):
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    if np.any(row_sums == 0):
        raise ValueError("At least one letter never appears; cannot compute percentages.")
    return (counts / row_sums) * 100


def plot_heatmap(percentages, title, save_path: Path):
    """Plot heatmap with variants on the vertical axis."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_data = percentages  # rows=variants, cols=positions
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=positions,
        yticklabels=VARIANTS,
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        linecolor="black",
    )
    plt.title(title)
    plt.xlabel("Ranking Position")
    plt.ylabel("AI-Generated Variant")
    # Adjust annotation text color for contrast on the viridis colormap
    vmax = heatmap_data.max()
    threshold = vmax * 0.5 if vmax else 0
    for text in ax.texts:
        text.set_text(text.get_text() + "%")
        try:
            val = float(text.get_text().strip("%"))
        except ValueError:
            val = 0
        # viridis is dark at low values, bright at high values
        text.set_color("black" if val > threshold else "white")
    plt.tight_layout()
    frontiers_style.save(save_path)
    plt.close()


all_blocks = load_blocks(FILE_PATH)

# Overall heatmap
overall_percent = to_percentages(tally_blocks(all_blocks))
plot_heatmap(overall_percent, "Placement Percentage (All Raters)", OUTPUT_DIR / "overall.tif")

# Split: first 5 blocks (male) vs remaining (female)
male_blocks = all_blocks[:5]
female_blocks = all_blocks[5:]

if male_blocks:
    male_percent = to_percentages(tally_blocks(male_blocks))
    plot_heatmap(male_percent, "Placement Percentage (Male)", OUTPUT_DIR / "male.png")

if female_blocks:
    female_percent = to_percentages(tally_blocks(female_blocks))
    plot_heatmap(female_percent, "Placement Percentage (Female)", OUTPUT_DIR / "female.png")
