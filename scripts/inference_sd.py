"""Generate a surgical video frame from a kinematics vector + gesture label.

Loads a checkpoint produced by ``train_sd.py``, runs DDIM denoising
conditioned on the kinematic embedding, and saves the result as a PNG.

This project supports only the JIGSAWS ``suturing`` task.

Example::

    python scripts/inference_sd.py \\
        --checkpoint ./checkpoints/suturing_expert_lora/step_1480.pt \\
        --kinematics_file /data/JIGSAWS/suturing/kinematics/allgestures/Suturing_B001.txt \\
        --gesture_label 3 \\
        --output_path ./generated_frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from suturing_pipeline.data.data_utils import parse_kinematics
from suturing_pipeline.synthesis.sd_sampler import SDSampler


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a frame conditioned on JIGSAWS kinematics."
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to a train_sd.py checkpoint."
    )
    p.add_argument(
        "--model_id",
        default=None,
        help="HuggingFace model ID.  If omitted, read from checkpoint args.",
    )
    p.add_argument(
        "--kinematics_file",
        required=True,
        help="Path to a JIGSAWS kinematics .txt file.",
    )
    p.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Which row (frame) of the kinematics file to use (default 0).",
    )
    p.add_argument(
        "--gesture_label",
        type=int,
        required=True,
        help="Integer gesture class index (0-indexed).",
    )
    p.add_argument("--output_path", default="./generated_frame.png")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--or_ambience",
        action="store_true",
        help=(
            "No effect for single-frame inference. For AudioGen OR ambience under "
            "Bark, run scripts/generate_eval_video.py with --enable_narration "
            "and this same flag."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if getattr(args, "or_ambience", False):
        print(
            "Note: --or_ambience is ignored here (no audio track). "
            "Use scripts/generate_eval_video.py --enable_narration --or_ambience."
        )

    sampler = SDSampler(
        checkpoint_path=args.checkpoint,
        model_id=args.model_id,
        device=args.device,
        image_size=args.image_size,
    )
    print(f"Loaded SD on device: {sampler.device}")

    kin_all = parse_kinematics(args.kinematics_file)
    if args.frame_index >= len(kin_all):
        raise IndexError(
            f"frame_index {args.frame_index} out of range "
            f"(file has {len(kin_all)} rows)"
        )
    kin_row = kin_all[args.frame_index]

    print(
        f"Running DDIM denoising ({args.num_inference_steps} steps) "
        f"for gesture={args.gesture_label} ..."
    )
    image = sampler.sample(
        kin_row=kin_row,
        gesture_int=args.gesture_label,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Saved generated frame to {out_path}")


if __name__ == "__main__":
    main()
