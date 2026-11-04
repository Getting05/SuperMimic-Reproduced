# ruff: noqa: I001
"""Run explicitly with python or torchrun; uses temporary outputs and small batches.

SGS_SMOKE_LIBRARY=/path/to/compact.pt python tests/sgs_smoke.py
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 tests/sgs_smoke.py
"""

import os
from pathlib import Path

# Apply the normal GPU pinning entry point before importing torch.
import supermimic.scripts.train  # noqa: F401
import torch

from supermimic.config import TrainConfig
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.training.loop import Trainer

cfg = TrainConfig()
cfg.env.robot = "p2"
cfg.env.sampling_mode = "sgs"
cfg.env.num_envs = 2
cfg.env.replay_capacity = 128
cfg.env.obs_noise = False
cfg.env.reset_noise = False
cfg.env.domain_randomization = False
cfg.data.library = Path(os.environ["SGS_SMOKE_LIBRARY"])
cfg.log_root = Path(os.environ["SGS_SMOKE_OUTPUT"])
cfg.run_name = "training"
cfg.schedule.total_iterations = 52
cfg.schedule.warmup_iterations = 12
cfg.schedule.updates_per_env_step = .05
cfg.schedule.log_interval = 26
cfg.wm_trainer.batches_per_update = 1
cfg.wm_trainer.batch_size = 8
cfg.wm_trainer.compile = None
cfg.policy_trainer.batch_size = 8
cfg.policy_trainer.horizon = 8
cfg.policy_trainer.critic_epochs = 1
cfg.policy_trainer.critic_minibatches = 1
cfg.policy_trainer.compile = None
cfg.policy.fsq = True
checkpoint = cfg.log_root / "training" / "ckpt_latest.pt"
if os.environ.get("SGS_SMOKE_PHASE") == "resume":
    cfg.resume = checkpoint
    cfg.run_name = "resume"
    cfg.schedule.total_iterations = 54
trainer = Trainer(cfg)
if cfg.resume is not None:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = torch.load(checkpoint.parent / ck["sgs_rank_files"][trainer.rank],
                       map_location="cpu", weights_only=False)
    for name, value in trainer.command.sgs_sampler.state_dict().items():
        assert torch.equal(value.cpu(), state["sampler"][name]), name
    assert trainer.start_iteration == 52
trainer.run()
assert trainer.num_updates >= 1
assert trainer.command.sgs_sampler.total_updates >= 1
saved_sampler = {name: value.clone() for name, value in trainer.command.sgs_sampler.state_dict().items()}
trainer.env.close()
print("SGS_TRAIN_OK", flush=True)

# Distributed resume uses a second torchrun invocation, avoiding process-group
# destruction/reinitialization in one interpreter.
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    raise SystemExit(0)

cfg.resume = checkpoint
cfg.run_name = "resume"
cfg.schedule.total_iterations = 54
resumed = Trainer(cfg)
for name, value in resumed.command.sgs_sampler.state_dict().items():
    assert torch.equal(value, saved_sampler[name]), name
assert resumed.start_iteration == 52
resumed.run()
resumed.env.close()
print("SGS_TRAIN_RESUME_OK", flush=True)

# Single-process integration checks the exact 50-step timeout and ReplayBuffer.
if int(os.environ.get("WORLD_SIZE", "1")) == 1:
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(cfg.data.library)], robot="p2", sampling_mode="sgs", num_envs=2,
        compact_library=True, replay_capacity=128, root_height_threshold=100.,
        root_tilt_threshold=100., obs_noise=False, reset_noise=False, domain_randomization=False,
        sgs_root_pos_threshold=100., sgs_body_pos_threshold=100.,
    ))
    cmd = env.command_manager.get_term("motion")
    env.reset()
    for step in range(1, 51):
        obs, reward, term, timeout, _ = env.step(torch.zeros(2, 29, device=env.device))
        assert not term.any()
        assert bool(timeout.all()) == (step == 50)
        assert int(cmd.sgs_sampler.total_updates) == (2 if step == 50 else 0)
    rec = env.recorder_manager.get_term("replay")
    windows = rec.buffer.sample_windows(8, 8)
    assert torch.isfinite(windows.states).all()
    env.close()
    print("SGS_50_STEP_REPLAY_OK", flush=True)
