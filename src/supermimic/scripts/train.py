"""Train SuperMimic.  Example:

  uv run sm-train --run-name e1_walk --data.clips walk1_subject1 --env.num-envs 1024
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

# Preserve the launcher's resource inventory before torchrun narrows rank affinity.
os.environ.setdefault("SM_EVAL_ALLOWED_CPU_IDS", ",".join(map(str, sorted(os.sched_getaffinity(0)))))
os.environ.setdefault("SM_EVAL_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

# Under torchrun, pin each rank to its own GPU *before* CUDA/warp initialise: MuJoCo Warp and
# mjlab's simulation always run on the first visible device, so cfg.device alone is not enough.
# Skipped on a box with fewer GPUs than ranks (single-GPU smoke tests share cuda:0 via gloo).
def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


# A preset CUDA_VISIBLE_DEVICES list (e.g. "0,2,3" to skip a bad card) is indexed by LOCAL_RANK.
_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
if "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail) and (not _preset or len(_preset) > 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"
    # Give each rank its own slice of cores (the training loop is single-thread CPU bound;
    # sharing cores between ranks adds jitter that lockstep all-reduces turn into waiting).
    try:
        cores = sorted(os.sched_getaffinity(0))
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local
        if per >= 2:
            r = int(os.environ["LOCAL_RANK"])
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
    except (OSError, ValueError):
        pass

import tyro

from supermimic.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    known, remaining = parser.parse_known_args()
    from supermimic.config_io import load_config
    defaults = load_config(known.config, TrainConfig()) if known.config else TrainConfig()
    cfg = tyro.cli(TrainConfig, default=defaults, args=remaining)
    if cfg.evaluation.enabled:
        if cfg.evaluation.interval < 1:
            raise ValueError("evaluation.interval must be positive")
        from supermimic.eval.config import EvalConfig
        settings = load_config(cfg.evaluation.config_file, EvalConfig()) if cfg.evaluation.config_file else cfg.evaluation.settings
        settings.validate()
    from supermimic.training.loop import Trainer

    trainer = Trainer(cfg)
    if cfg.resume is None and os.environ.get("SM_SAVE_INITIAL_CHECKPOINT") == "1":
        trainer.save("ckpt_initial.pt")
    trainer.run()


if __name__ == "__main__":
    main()
