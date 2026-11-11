# Unified checkpoint evaluation

Run commands from the repository root with its Python environment active.
Training configs live in `configs/train/`; evaluation configs in `configs/eval/`.
Both TOML and JSON are accepted. Paths in configs are relative to the working
directory, not the config file. Unknown keys and invalid resource settings fail
before simulation starts. The example data/checkpoint paths must be edited.

## Standalone evaluation

```bash
python scripts/evaluate.py --config configs/eval/cpu.toml
python scripts/evaluate.py --config configs/eval/gpu.toml \
  --checkpoint logs/my_run/ckpt_0014000.pt --output-dir evals/my_run_14000
python scripts/evaluate.py --list-metrics
```

`checkpoint` selects weights. Architecture, observation layout, robot, actuator
mode and future-reference settings come from the checkpoint's saved config.
93D/99D layouts are restored automatically, including older parity checkpoints
without `policy_obs_ref_rot6d`. Old pre-parity dev/FSQ checkpoints are rejected
because their state layout and model contract differ.

`checkpoint_config` optionally names a saved training TOML/JSON config. Its
provided `policy` and `env` fields are checked against the saved checkpoint;
it cannot silently override the model contract. Partial files are allowed.

### Evaluation range

Each `[[datasets]]` specifies a named scope such as train, val, test or a custom
subset. `path` is a packed `.pt` library or an NPZ directory. `format` is `auto`,
`packed` or `npz`. `clips` and `clip_list` restrict motion names; `limit` takes the
first selected clips **before worker sharding**. NPZ also supports the existing
CSV `manifest` filter. A packed file remains one source; for several packed
shards add separate named dataset entries, e.g. `train_rank00`, `train_rank01`.
Results include per-scope summaries and one combined summary.

### Resources

| Field under `[resources]` | Meaning |
| --- | --- |
| `device` | `cpu` or `cuda` |
| `gpu_ids` | Physical GPU indices, e.g. `[7]` or `[6,7]`; independent of the training rank's GPU mask |
| `workers` | Parallel processes; GPU mode expands to at least one worker per GPU; larger counts must be a multiple of GPU count |
| `num_envs` | Parallel simulator environments **per worker** |
| `threads_per_worker` | Torch/BLAS thread limit for each worker |
| `cpu_ids` | Explicit logical CPU indices, e.g. `[96,97,98,99]` |
| `cpu_count` | Use this many allowed logical CPUs |
| `cpu_fraction` | Fraction of available CPU capacity, accounting for cgroup quota, e.g. `0.70` |
| `library_device` | Store references in `cpu` memory or on the worker's `gpu` |

Choose only one of `cpu_ids`, `cpu_count`, `cpu_fraction`. CPU budget must cover
`workers * threads_per_worker`. Workers receive disjoint affinity sets; this is
an affinity/thread budget, not a promise of a fixed live utilization percentage.
Each GPU worker sees its selected card as `cuda:0`. CPU workers hide CUDA.
Workers remove inherited torchrun group variables and start in fresh processes.
Motion queues are disjoint across workers and are merged automatically.
Packed libraries are loaded by each worker, so account for RAM use when choosing
many workers on a large library. GPU IDs are interpreted using PCI bus order,
consistent with the existing docker150 launch scripts.

## Evaluation during training

```bash
python -m supermimic.scripts.train --config configs/train/p2_fsq.toml \
  --evaluation.enabled --evaluation.config-file configs/eval/cpu.toml
```

The same CLI works under the existing `torchrun` launch. Existing CLI flags
override the training file. Automatic evaluation is disabled by default.

```toml
[evaluation]
enabled = true
interval = 4000           # training update iterations
on_start = false
on_finish = true
wait_on_finish = true
config_file = "configs/eval/cpu.toml"
```

Alternatively omit `config_file` and put evaluation fields directly in the
training file under `[evaluation.settings]`, `[[evaluation.settings.datasets]]`,
`[evaluation.settings.resources]`, `[evaluation.settings.quality]`, etc.
When `config_file` is set it replaces `settings`. The checkpoint and output path
are supplied by training, regardless of the corresponding template values.

All ranks save an immutable evaluation checkpoint; rank zero starts a separate
evaluation controller. Training continues while evaluation runs. Only one
automatic evaluation is active per run; interval triggers while busy are skipped
and recorded. With `wait_on_finish=true`, training waits for any active evaluation
then evaluates the final snapshot. With `false`, it returns immediately; an active
job continues independently, and a busy final trigger is skipped.

Artifacts live in `logs/<run>/eval/<checkpoint-stem>/`, with the resolved config,
controller/worker logs, progress and results. Trigger history is in
`eval/automatic_status.json`. Evaluation errors are recorded without terminating
the training process. Initial/final/interval snapshots remain available for
reproducibility. Choose a free GPU or a CPU budget appropriate for concurrent
training; the evaluator does not reserve hardware against unrelated jobs.

## Protocol and metric options

Every selected clip starts from its first reference frame. Observation/action
noise, reset noise and domain randomization are disabled. Actuator mode follows
the checkpoint; `actuator_delay` fixes a supported delay (default 2 simulation
steps, matching the earlier CPU heldout evaluations).

Core metrics are always recorded: success/failure, failure reason, executed steps,
completion, full-reference-horizon `norm_return`, and executed-steps
`executed_norm_return`. The latter does not penalize the unexecuted remainder.
The denominator of full-horizon return is `max_step_reward * (frames - 1)`.
Natural clip-end timeout is successful; a physical root-height/root-tilt failure
always takes precedence, including on the last step. The command's clip-end
condition can end at `frames - 2`, so completion uses that effective horizon and
is capped at 1. Original frame and step counts remain in every record.

Set `metrics = [...]` to choose state-error metrics, or `[]` for core metrics only.

| Metric key | Quantity | Unit |
| --- | --- | --- |
| `pelvis_position` | Pelvis Euclidean position error | m |
| `pelvis_orientation` | Pelvis quaternion angular error | rad |
| `body_position_world` | Mean tracked-body world position error | m |
| `body_orientation_world` | Mean tracked-body world angular error | rad |
| `body_position_reanchored` | Body position error after reference translation/yaw reanchoring | m |
| `body_orientation_reanchored` | Body angular error after reference reanchoring | rad |
| `body_position_root_relative` | Body position error relative to each trajectory's root (world axes) | m |
| `joint_position` | Joint position MAE over the correct Vibed71 `[13:42]` slice | rad |
| `joint_velocity` | Joint velocity MAE | rad/s |

State errors are averaged over **executed** steps. Summaries provide clip means
and executed-frame-weighted means. Errors for the unexecuted remainder are not
fabricated; compare errors together with completion/success/full-horizon return.
`reward_terms=true` records the environment's per-term weighted, dt-scaled rewards
and their executed-step means. The current terms cover body position/orientation/
linear velocity/angular velocity, root position/orientation, action rate and joint
limits. No contact/foot-sliding or motion smoothness metric is claimed here.

Quality filtering is optional:

```toml
[quality]
enabled = true
min_completion = 0.98
[quality.max_errors]
pelvis_position = 0.30
body_position_root_relative = 0.25
joint_position = 0.6981317007977318 # 40 degrees in radians
```

Threshold metrics are calculated even if absent from `metrics`. A physical failure
or quality flag makes the clip an exclusion candidate. Filtering writes lists;
it does not modify the motion dataset.

## Outputs and older entry points

`results.json` contains the resolved config, metric units, per-dataset/overall
summaries and per-clip records. `clips.csv` flattens metrics and reward terms.
`failures.json/.txt`, `low_quality.json/.txt`, `exclude_candidates.json/.txt`
provide records and unique IDs including scope/index/name. `status.json` reports
running/completed/failed, and `workers_*/` retains worker logs and shards.

`eval_benchmark.py`, `eval_packed_checkpoint.py` and `heldout_cpu_eval.py` now route
to this engine; their old invocation arguments remain supported, while output
uses the unified schema. Multi-GPU evaluation already merges worker outputs, so
manual shard merging is unnecessary. `eval_push.py` remains a separate G1-only
robustness experiment. Historical scripts under `logs/`/`evals/` are preserved;
new training no longer needs their watcher processes or hardcoded experiment
plans. Historical results are not rewritten. Earlier screening joint-error
results used the pre-parity slice and must not be treated as the corrected metric.
