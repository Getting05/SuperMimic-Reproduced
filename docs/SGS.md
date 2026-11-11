# Success Guided Sampling

SGS follows *A Balanced Data Diet: Addressing the Exploration Bottleneck in
Mega-Scale RL for Robot Control* (CoRL 2026). The locomotion defaults are target
0.66, concentration 5, temperature 2, epsilon 1e-8 and a 100-result rolling
Boolean history. Unvisited bins start at success rate zero. All valid bins
initially have equal probability. Sampling uses the beta kernel, an epsilon
floor and temperature softmax. The original return, uniform and start modes
remain selectable.

## Tracking success

Each bin denotes a reference-start configuration. SGS chooses a bin and a
uniform valid frame inside it. A start must leave 50 frames including the start
frame in the same clip. Bins without such a start are excluded and counted.
The reference does not advance on the reset call, so the 50th action is evaluated
against start + 49.

A simulator episode succeeds when it completes exactly 50 policy steps, has no
failure termination, and its mean root position distance and mean body position
distance are both below 0.15 m. Distances compare world-space simulator link
origins with the reference at each step, without root alignment. Body distance
is averaged over the configured tracked bodies. These thresholds and horizon
are SuperMimic task definitions, exposed as env.sgs_* options. They are separate
from the paper's sampling hyperparameters. Rewards are unchanged.

The SGS timeout term accumulates errors during termination evaluation, before
scene reset. Command reset reads the old termination flags before the termination
manager clears them. Every completed episode updates only its starting bin.
Manual resets without a termination discard unfinished outcomes. Forced-frame
evaluation does not update SGS. Stable sorting makes simultaneous updates to the
same ring deterministic in environment-index order, retaining the latest H.
ReplayBuffer sampling and neural networks/losses are unchanged.

## Training

P2 advanced actuators now default to false. Enable with
`--env.p2-advanced-actuators` when needed.

Single GPU, using the existing 15-clip compact library on docker150:

```bash
cd /root/code/SuperMimic-Reproduced
source .venv/bin/activate-docker150
CUDA_VISIBLE_DEVICES=0 sm-train \
  --env.robot p2 --policy.fsq --env.sampling-mode sgs \
  --data.library data/motions/lafan1_mix5clean_v2_p2_supermimic_50hz.pt \
  --env.num-envs 4096 --run-name p2_fsq_sgs \
  --wandb.enabled --wandb.project supermimic
```

Four GPUs, each loading the same library:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --standalone --nproc-per-node=4 \
  -m supermimic.scripts.train \
  --env.robot p2 --policy.fsq --env.sampling-mode sgs \
  --data.library data/motions/lafan1_mix5clean_v2_p2_supermimic_50hz.pt \
  --env.num-envs 4096 --run-name p2_fsq_sgs_4gpu \
  --wandb.enabled --wandb.project supermimic
```

Each rank maintains its own local success history. Startup compares dataset
signatures across ranks. Use one common complete library, including identical
clip order, for SGS. The earlier launch script that assigns different rank-local
motion subsets needs a common library argument for this mode.

Checkpoints have a main model file plus generation-specific `.rankN.sgs.pt`
sidecars. Save is collective. Copy the main file and all referenced sidecars
together. Sidecars are published before the main file and contain rank-local
history, counters and RNG state. Resume requires the same world size, file
content/order, bin size and SGS parameters. A legacy non-SGS checkpoint supplied
with `--resume` initializes model weights with fresh optimizers, counters and SGS
history. Simulator episodes and ReplayBuffer are refilled on resume.

## Metrics and verification

TensorBoard and optional W&B record the same scalar keys. `sgs/*` includes the
success-rate distribution, valid/excluded/unvisited bin counts, sampling entropy,
zero-success ratio, interval episode success rate, actual root/body errors and
total outcomes. Histograms also aggregate valid-bin success rates by clip.
Distributed logs describe rank 0's local sampler. No per-step cross-rank sampler
synchronization is performed.

```bash
python -m pytest tests/test_sgs_sampler.py -q -s
SGS_SMOKE_LIBRARY=/absolute/path/to/compact.pt \
SGS_SMOKE_OUTPUT=/tmp/sgs-single python tests/sgs_smoke.py
CUDA_VISIBLE_DEVICES=0,1 SGS_SMOKE_LIBRARY=/absolute/path/to/compact.pt \
SGS_SMOKE_OUTPUT=/tmp/sgs-two-rank \
  torchrun --standalone --nproc-per-node=2 tests/sgs_smoke.py
```

For distributed resume, repeat the torchrun command with `SGS_SMOKE_PHASE=resume`
and the same SGS_SMOKE_OUTPUT.

The smoke script uses two environments, small batches and short horizons for
training checks, then exercises the standard 50-step reset horizon and replay
windows in the single-rank case. It writes only under SGS_SMOKE_OUTPUT.


## Optional rank-sharded SGS for DDP

By default, all ranks in an SGS DDP run must load an identical MotionLibrary.
When training with independent packed files `rank_00.pt`, `rank_01.pt`, ...
instead, enable `--env.sgs-rank-sharded` (TOML: `[env] sgs_rank_sharded = true`).
Each rank keeps its own SGS bins and sampling statistics; only model parameters
are synchronized through DDP. Files must be named in rank order and have
different contents. Mixed/repeated libraries are rejected.

The legacy `SUPERMIMIC_SGS_RANK_SHARDED=1` environment variable remains
accepted for older launch scripts. Leave sharding disabled for shared datasets.
Full-state checkpoints already store per-rank sampler state for correct resume.
