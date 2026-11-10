import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def percentile_normalize(attn_scores, low=0, high=100):
    low_val = np.percentile(attn_scores, low)
    high_val = np.percentile(attn_scores, high)

    normed = (attn_scores - low_val) / (high_val - low_val + 1e-8)
    normed = np.clip(normed, 0.0, 1.0)
    return normed


def generate_heatmap(wsi_path, h5_path, prob_path, save_root, downsample=32, patch_size=512):
    import openslide

    slide = openslide.OpenSlide(wsi_path)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((W // downsample, H // downsample))
    thumb = thumb.convert("RGB")

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    probs = np.load(prob_path)
    if probs.shape[0] == 1:
        probs = probs[0]
    elif probs.ndim == 2:
        probs = probs[:, 1]

    norm_probs = percentile_normalize(probs)
    heatmap = np.zeros((H // downsample, W // downsample), dtype=np.float32)

    patch_down = patch_size // downsample
    for (x, y), p in zip(coords, norm_probs):
        x_down = x // downsample
        y_down = y // downsample
        heatmap[y_down:y_down + patch_down, x_down:x_down + patch_down] = p
    cmap = plt.get_cmap("jet")
    heatmap_color = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    heatmap_img = Image.fromarray(heatmap_color).resize(thumb.size)
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    heatmap_img.save(os.path.join(save_root, f"{slide_id}_heatmap.jpg"))

    print(f"[OK] {slide_id} finished")


def main():
    parser = argparse.ArgumentParser(description="Render patch-probability heatmaps for a folder of WSIs.")
    parser.add_argument("--wsi_dir", required=True, help="folder with the .tif slides")
    parser.add_argument("--h5_dir", required=True, help="folder with {slide_id}.h5 (patch coordinates)")
    parser.add_argument("--prob_dir", required=True, help="folder with {slide_id}.npy (patch probabilities)")
    parser.add_argument("--out_dir", required=True, help="folder that receives {slide_id}_heatmap.jpg")
    parser.add_argument("--csv", default=None,
                        help="optional csv with slide_id/label columns; slides labelled "
                             "'Solid Tissue Normal' are skipped")
    parser.add_argument("--downsample", type=int, default=32, help="thumbnail downsample factor (default: 32)")
    parser.add_argument("--patch_size", type=int, default=512, help="patch size in pixels (default: 512)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    skip_ids = set()
    if args.csv:
        df = pd.read_csv(args.csv)
        skip_ids = set(df.loc[df["label"] == "Solid Tissue Normal", "slide_id"])

    for fname in os.listdir(args.wsi_dir):
        if not fname.endswith(".tif"):
            continue
        slide_id = os.path.splitext(fname)[0]
        if slide_id in skip_ids:
            continue

        wsi_path = os.path.join(args.wsi_dir, fname)
        h5_path = os.path.join(args.h5_dir, f"{slide_id}.h5")
        prob_path = os.path.join(args.prob_dir, f"{slide_id}.npy")

        if os.path.exists(h5_path) and os.path.exists(prob_path):
            generate_heatmap(wsi_path, h5_path, prob_path, args.out_dir, args.downsample, args.patch_size)
        else:
            print(f"{slide_id}: files not found")


if __name__ == "__main__":
    main()
