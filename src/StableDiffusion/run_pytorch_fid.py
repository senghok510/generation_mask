from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_torch_home() -> Path:
    return Path(__file__).resolve().parents[2] / ".torch_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FID using the pytorch-fid package used by the paper."
    )
    parser.add_argument("--real-dir", required=True, type=Path, help="Directory containing real images.")
    parser.add_argument("--fake-dir", required=True, type=Path, help="Directory containing generated images.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size passed to pytorch-fid.")
    parser.add_argument("--device", default="cuda:0", help="Torch device string.")
    parser.add_argument("--dims", type=int, default=2048, help="Feature dimensionality for FID.")
    parser.add_argument("--num-workers", type=int, default=0, help="Worker count passed to pytorch-fid.")
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=default_torch_home(),
        help="Directory used by torch/torchvision to cache the Inception weights.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path to write the result as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())

    if not args.real_dir.exists():
        raise FileNotFoundError(f"Real directory does not exist: {args.real_dir}")
    if not args.fake_dir.exists():
        raise FileNotFoundError(f"Fake directory does not exist: {args.fake_dir}")

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise SystemExit(
            "pytorch-fid is not installed in the active interpreter. "
            "Install it in the shared venv, then rerun this command."
        ) from exc

    fid_value = calculate_fid_given_paths(
        [str(args.real_dir.resolve()), str(args.fake_dir.resolve())],
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        num_workers=args.num_workers,
    )

    result = {
        "real_dir": str(args.real_dir.resolve()),
        "fake_dir": str(args.fake_dir.resolve()),
        "batch_size": args.batch_size,
        "device": args.device,
        "dims": args.dims,
        "num_workers": args.num_workers,
        "fid": float(fid_value),
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
