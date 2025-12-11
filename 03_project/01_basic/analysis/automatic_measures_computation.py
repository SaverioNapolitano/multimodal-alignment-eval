import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_msssim import ms_ssim
import lpips
import clip

# Root directory containing folders 01..15 and this script
BASE_DIR = Path(__file__).parent
OUTPUT_CSV = BASE_DIR / "automatic_measures_results.csv"

# Accept common image extensions
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
VARIANTS = ["a", "b", "c", "d"]


def find_image(folder: Path, stem: str) -> Optional[Path]:
    """Return the first matching image path for the given stem."""
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_pil(path: Path, target_size=None) -> Image.Image:
    """Load an image as RGB, optionally resizing to target_size (W, H)."""
    img = Image.open(path).convert("RGB")
    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.BICUBIC)
    return img


def tensor_for_lpips(img: Image.Image, device):
    """Convert PIL image to tensor in [-1, 1] for LPIPS."""
    t = T.ToTensor()(img) * 2 - 1  # [0,1] -> [-1,1]
    return t.unsqueeze(0).to(device)


def tensor_for_msssim(img: Image.Image, device):
    """Convert PIL image to tensor in [0,1] for MS-SSIM."""
    t = T.ToTensor()(img).unsqueeze(0).to(device)
    return t


def compute_metrics_for_pair(
    ref_img: Image.Image,
    gen_img: Image.Image,
    lpips_model,
    device,
    clip_model,
    clip_preprocess,
):
    """Compute LPIPS, MS-SSIM, and CLIP cosine similarity between two PIL images."""
    # Ensure same size for perceptual metrics
    target_size = ref_img.size
    gen_img_resized = gen_img if gen_img.size == target_size else gen_img.resize(target_size, Image.BICUBIC)

    with torch.inference_mode():
        # LPIPS
        ref_lpips = tensor_for_lpips(ref_img, device)
        gen_lpips = tensor_for_lpips(gen_img_resized, device)
        lpips_score = lpips_model(ref_lpips, gen_lpips).item()

        # MS-SSIM
        ref_msssim = tensor_for_msssim(ref_img, device)
        gen_msssim = tensor_for_msssim(gen_img_resized, device)
        msssim_score = ms_ssim(ref_msssim, gen_msssim, data_range=1.0).item()

        # CLIP cosine similarity
        ref_clip = clip_preprocess(ref_img).unsqueeze(0).to(device)
        gen_clip = clip_preprocess(gen_img).unsqueeze(0).to(device)
        ref_feat = clip_model.encode_image(ref_clip).float()
        gen_feat = clip_model.encode_image(gen_clip).float()
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)
        clip_cos = torch.nn.functional.cosine_similarity(ref_feat, gen_feat).item()

        return lpips_score, msssim_score, clip_cos


def collect_folder_results(folder: Path, lpips_model, device, clip_model, clip_preprocess) -> List[dict]:
    """Compute metrics for one folder (e.g., 01) against all variants."""
    folder_name = folder.name
    ref_path = find_image(folder, f"{folder_name}_red") or find_image(folder, f"{folder_name}_ref")
    if not ref_path:
        raise FileNotFoundError(f"No reference image found in {folder} (looked for *_red or *_ref).")

    ref_img = load_pil(ref_path)
    results = []

    for variant in VARIANTS:
        gen_path = find_image(folder, f"{folder_name}_{variant}")
        if not gen_path:
            # Skip missing variants but continue processing the rest
            continue
        gen_img = load_pil(gen_path)
        lpips_score, msssim_score, clip_cos = compute_metrics_for_pair(
            ref_img, gen_img, lpips_model, device, clip_model, clip_preprocess
        )
        results.append(
            {
                "folder": folder_name,
                "variant": variant,
                "lpips": lpips_score,
                "ms_ssim": msssim_score,
                "clip_cosine": clip_cos,
                "reference": ref_path.name,
                "image": gen_path.name,
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Compute LPIPS, MS-SSIM and CLIP similarity for image variants.")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Force a device. Defaults to auto (cuda > mps > cpu).",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        help="Optional list of folder names (e.g., 01 02) to process. Defaults to all numeric folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help="Where to write the CSV (default: automatic_measures_results.csv next to the script).",
    )
    return parser.parse_args()


def choose_device(preferred: Optional[str]) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iter_target_folders(base_dir: Path, only: Optional[Sequence[str]] = None) -> Iterable[Path]:
    allowed = {name.zfill(2) for name in only} if only else None
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.isdigit():
            continue
        if allowed and folder.name not in allowed:
            continue
        yield folder


def main():
    args = parse_args()
    device = choose_device(args.device)
    device = 'cpu'
    print(f"Using device: {device}", file=sys.stderr)

    torch.set_grad_enabled(False)

    # Models
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "folder",
        "variant",
        "lpips",
        "ms_ssim",
        "clip_cosine",
        "reference",
        "image",
    ]

    rows_written = 0
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for folder in iter_target_folders(BASE_DIR, args.folders):
            print(f"Processing folder {folder.name}...", file=sys.stderr)
            try:
                folder_results = collect_folder_results(folder, lpips_model, device, clip_model, clip_preprocess)
            except Exception as exc:  # keep going if one folder fails
                print(f"Skipping folder {folder.name}: {exc}", file=sys.stderr)
                continue

            writer.writerows(folder_results)
            rows_written += len(folder_results)

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
