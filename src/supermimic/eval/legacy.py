"""Compatibility for the earlier packed evaluation commands."""
import argparse
import json
from pathlib import Path

from supermimic.eval.config import DatasetCfg, EvalConfig, ResourceCfg
from supermimic.eval.runner import main as configured_main, run


def packed_main(device):
    import sys
    if "--config" in sys.argv or "--list-metrics" in sys.argv:
        return configured_main()
    parser = argparse.ArgumentParser(description="Compatibility wrapper; prefer evaluate.py --config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--limit-clips", type=int)
    args = parser.parse_args()
    cfg = EvalConfig(checkpoint=args.checkpoint, output_dir=args.out.parent / (args.out.stem + "_details"),
                     datasets=[DatasetCfg(path=args.library, format="packed", limit=args.limit_clips)],
                     resources=ResourceCfg(device=device, num_envs=args.num_envs))
    if device == "cuda":
        import os
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
        cfg.resources.gpu_ids = [int(visible)] if visible else [0]
    result = run(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
