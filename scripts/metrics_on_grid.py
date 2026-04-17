"""Compute PSNR / SSIM / color-histogram metrics on eval_grid1.png.

Crops the 4 (real, generated) pairs from the matplotlib grid and reports
per-sample metrics plus aggregates.  Also reports "baseline" metrics that
compare each generated image against a DIFFERENT real frame, so we can
tell whether the model's outputs are merely capturing the scene's
average statistics or actually tracking the conditioning signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn


GRID_PATH = Path(
    "/Users/amyngo/SS-SD/outputs/eval/smoke_test/eval_grid1.png"
)

# The matplotlib figure was saved with figsize=(6, 3*n) at dpi=120 +
# bbox_inches="tight".  The tight bbox trims whitespace, so we detect
# the 8 image tiles directly rather than hard-coding pixel offsets.


def _find_tile_rows(arr: np.ndarray) -> list[tuple[int, int]]:
    """Return row spans [(y0, y1), ...] for each image row in the grid.

    The matplotlib background is white (~255).  Image tiles are clearly
    darker (the JIGSAWS scenes are dim).  We scan for runs of rows whose
    mean brightness drops well below the white background.
    """
    gray = arr.mean(axis=(1, 2))
    dark = gray < 230
    spans: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for y, d in enumerate(dark):
        if d and not in_run:
            in_run = True
            start = y
        elif not d and in_run:
            in_run = False
            if y - start > 60:
                spans.append((start, y))
    if in_run and len(dark) - start > 60:
        spans.append((start, len(dark)))
    return spans


def _find_tile_cols(arr: np.ndarray, y0: int, y1: int) -> list[tuple[int, int]]:
    """Inside a row band, return column spans for the 2 image tiles."""
    sub = arr[y0:y1]
    col_gray = sub.mean(axis=(0, 2))
    dark = col_gray < 230
    spans: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for x, d in enumerate(dark):
        if d and not in_run:
            in_run = True
            start = x
        elif not d and in_run:
            in_run = False
            if x - start > 60:
                spans.append((start, x))
    if in_run and len(dark) - start > 60:
        spans.append((start, len(dark)))
    return spans


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _hist_chi2(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Chi-square distance between per-channel histograms (lower = closer)."""
    total = 0.0
    for c in range(3):
        ha, _ = np.histogram(a[..., c], bins=bins, range=(0, 255), density=True)
        hb, _ = np.histogram(b[..., c], bins=bins, range=(0, 255), density=True)
        denom = ha + hb + 1e-12
        total += 0.5 * np.sum(((ha - hb) ** 2) / denom)
    return float(total)


def _edge_map(img: np.ndarray) -> np.ndarray:
    """Simple Sobel-magnitude edge map normalised to [0, 1]."""
    from scipy.ndimage import sobel

    gray = img.mean(axis=-1)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.sqrt(gx * gx + gy * gy)
    mx = mag.max()
    return mag / (mx + 1e-6)


def _edge_iou(a: np.ndarray, b: np.ndarray, thresh: float = 0.15) -> float:
    ea = _edge_map(a) > thresh
    eb = _edge_map(b) > thresh
    inter = np.logical_and(ea, eb).sum()
    union = np.logical_or(ea, eb).sum()
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    img = np.asarray(Image.open(GRID_PATH).convert("RGB"))
    rows = _find_tile_rows(img)
    print(f"Detected {len(rows)} row bands")

    pairs: list[dict] = []
    labels = [
        ("Suturing_C005 f=30",   "G1"),
        ("Suturing_C005 f=2520", "G11"),
        ("Suturing_C005 f=1320", "G2"),
        ("Suturing_C005 f=2100", "G3"),
    ]
    for i, (y0, y1) in enumerate(rows):
        cols = _find_tile_cols(img, y0, y1)
        if len(cols) < 2:
            continue
        x0a, x1a = cols[0]
        x0b, x1b = cols[1]
        real = img[y0:y1, x0a:x1a]
        gen = img[y0:y1, x0b:x1b]
        # Resize to common resolution via PIL for fair compare.
        size = (min(real.shape[1], gen.shape[1]), min(real.shape[0], gen.shape[0]))
        real = np.asarray(Image.fromarray(real).resize(size, Image.BILINEAR))
        gen = np.asarray(Image.fromarray(gen).resize(size, Image.BILINEAR))
        info = labels[i] if i < len(labels) else (f"pair_{i}", "?")
        pairs.append({
            "label": info[0],
            "gesture": info[1],
            "real": real,
            "gen": gen,
        })

    print(f"\nExtracted {len(pairs)} (real, gen) pairs")
    print(f"Tile size after resize: {pairs[0]['real'].shape}")

    print("\n" + "=" * 92)
    print(f"{'sample':<26} {'gesture':<6} {'PSNR':>7}  {'SSIM':>7}  {'Hist chi^2':>10}  {'Edge IoU':>9}")
    print("-" * 92)
    per_sample_psnr = []
    per_sample_ssim = []
    per_sample_hist = []
    per_sample_edge = []
    records: list[dict] = []
    for p in pairs:
        real, gen = p["real"], p["gen"]
        ps = _psnr(real, gen)
        ss = float(ssim_fn(real, gen, channel_axis=-1, data_range=255))
        h = _hist_chi2(real, gen)
        e = _edge_iou(real, gen)
        per_sample_psnr.append(ps)
        per_sample_ssim.append(ss)
        per_sample_hist.append(h)
        per_sample_edge.append(e)
        records.append({
            "sample": p["label"],
            "gesture": p["gesture"],
            "psnr_db": round(ps, 2),
            "ssim": round(ss, 4),
            "hist_chi2": round(h, 4),
            "edge_iou": round(e, 4),
        })
        print(f"{p['label']:<26} {p['gesture']:<6} {ps:>6.2f}   {ss:>6.4f}  {h:>10.4f}  {e:>9.4f}")

    print("-" * 92)
    print(
        f"{'MEAN':<26} {'':<6} {np.mean(per_sample_psnr):>6.2f}   "
        f"{np.mean(per_sample_ssim):>6.4f}  "
        f"{np.mean(per_sample_hist):>10.4f}  "
        f"{np.mean(per_sample_edge):>9.4f}"
    )

    # ---- baseline: compare gen[i] to real[j], j != i -------------------
    print("\nCross-pair baseline (gen_i vs real_j, j != i):")
    print("If the model is truly kinematic-conditioned, the diagonal (matched")
    print("pair) metrics should be NOTICEABLY better than off-diagonal ones.")
    print("If the two are indistinguishable, the model is just painting a")
    print("scene-average image and ignoring the conditioning.\n")
    n = len(pairs)
    psnr_mat = np.zeros((n, n))
    ssim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            psnr_mat[i, j] = _psnr(pairs[j]["real"], pairs[i]["gen"])
            ssim_mat[i, j] = ssim_fn(
                pairs[j]["real"], pairs[i]["gen"],
                channel_axis=-1, data_range=255,
            )

    print("PSNR matrix  rows = gen_i, cols = real_j  (dB)")
    header = "           " + " ".join(f"real_{j}".rjust(8) for j in range(n))
    print(header)
    for i in range(n):
        row = " ".join(f"{psnr_mat[i, j]:>8.2f}" for j in range(n))
        print(f"gen_{i}     {row}")

    print("\nSSIM matrix  rows = gen_i, cols = real_j")
    print(header)
    for i in range(n):
        row = " ".join(f"{ssim_mat[i, j]:>8.4f}" for j in range(n))
        print(f"gen_{i}     {row}")

    diag_psnr = np.diag(psnr_mat).mean()
    off_psnr = (psnr_mat.sum() - np.diag(psnr_mat).sum()) / (n * n - n)
    diag_ssim = np.diag(ssim_mat).mean()
    off_ssim = (ssim_mat.sum() - np.diag(ssim_mat).sum()) / (n * n - n)
    print(f"\nMean diagonal PSNR  = {diag_psnr:.2f} dB  "
          f"| off-diagonal = {off_psnr:.2f} dB  "
          f"| delta = {diag_psnr - off_psnr:+.2f} dB")
    print(f"Mean diagonal SSIM  = {diag_ssim:.4f}    "
          f"| off-diagonal = {off_ssim:.4f}    "
          f"| delta = {diag_ssim - off_ssim:+.4f}")

    out_path = GRID_PATH.parent / "metrics.json"
    out_path.write_text(
        json.dumps({
            "per_sample": records,
            "mean_psnr_db": round(float(np.mean(per_sample_psnr)), 3),
            "mean_ssim": round(float(np.mean(per_sample_ssim)), 4),
            "mean_hist_chi2": round(float(np.mean(per_sample_hist)), 4),
            "mean_edge_iou": round(float(np.mean(per_sample_edge)), 4),
            "cross_pair": {
                "psnr_matrix_db": psnr_mat.round(3).tolist(),
                "ssim_matrix": ssim_mat.round(4).tolist(),
                "diag_minus_off_psnr_db": round(float(diag_psnr - off_psnr), 3),
                "diag_minus_off_ssim": round(float(diag_ssim - off_ssim), 4),
            },
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
