#!/usr/bin/env python3
"""Fréchet Inception Distance (FID) between two image folders.

Typical use after ``generate_eval_grid.py``: split exported PNGs into a
``real/`` folder and a ``generated/`` folder (same cardinality and pairing
order if you compare distributions fairly — for paired frames from the grid,
ensure filenames align).

Example::

    python scripts/compute_fid.py \\
        --path1 ./outputs/eval/run/real_frames \\
        --path2 ./outputs/eval/run/gen_frames \\
        --device cuda

Requires ``pytorch-fid`` (``pip install pytorch-fid``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute FID between two image directories (Inception-v3)."
    )
    p.add_argument(
        "--path1",
        required=True,
        help="First image directory (e.g. real frames).",
    )
    p.add_argument(
        "--path2",
        required=True,
        help="Second image directory (e.g. generated frames).",
    )
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        help="Inception pool3 layer dimension (default 2048).",
    )
    p.add_argument("--device", default=None, help="'cuda', 'cpu', or omit for auto.")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    p1 = Path(args.path1)
    p2 = Path(args.path2)
    if not p1.is_dir():
        sys.exit(f"Not a directory: {p1}")
    if not p2.is_dir():
        sys.exit(f"Not a directory: {p2}")

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as e:
        sys.exit(
            "pytorch-fid is required: pip install pytorch-fid\n" + str(e)
        )

    import torch

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid = calculate_fid_given_paths(
        [str(p1.resolve()), str(p2.resolve())],
        batch_size=args.batch_size,
        device=device,
        dims=args.dims,
        num_workers=args.num_workers,
    )
    print(f"FID ({p1.name} vs {p2.name}): {fid:.4f}")


if __name__ == "__main__":
    main()
