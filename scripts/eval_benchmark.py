"""NPZ benchmark compatibility CLI; use evaluate.py --config for new runs."""
from pathlib import Path
import json
import os

from supermimic.eval.config import DatasetCfg, EvalConfig, ResourceCfg
from supermimic.eval.runner import run


def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None,
         clip_list: Path | None = None, manifest: Path | None = None, num_envs: int = 1024,
         device: str = "cuda:0", out: Path | None = None):
    target = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    gpu_ids = [int(device.split(":")[-1])] if device.startswith("cuda:") else [0]
    visible = [i for i in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if i]
    if device.startswith("cuda") and visible:
        gpu_ids = [int(visible[gpu_ids[0]])]
    cfg = EvalConfig(checkpoint=ckpt, output_dir=target.parent / (target.stem + "_details"),
                     datasets=[DatasetCfg(path=npz_dir, format="npz", clips=clips or [], clip_list=clip_list, manifest=manifest)],
                     resources=ResourceCfg(device="cuda" if device.startswith("cuda") else "cpu", gpu_ids=gpu_ids, num_envs=num_envs))
    result = run(cfg)
    target.write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import sys
    if "--config" in sys.argv or "--list-metrics" in sys.argv:
        from supermimic.eval.runner import main as configured_main
        configured_main()
    else:
        import tyro
        tyro.cli(main)
