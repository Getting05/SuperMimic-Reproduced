# Vibed parity FSQ report

Branch: dev/vibed-parity-fsq.
Source: 0ad10dfb043077fb9c7536169503e74a60a960e4.
Pinned Vibed: 0525a23a2dcfa61f8543f98b84c075fe66d9d286.
Final SHA: the stage-10 commit on this branch (git rev-parse HEAD).
Ten implementation stages have separate commits. This report distinguishes
implemented behavior from unperformed acceptance checks.

## Solver decision

The user explicitly approved preserving actual source settings:
P2 nconmax128/njmax768, G1 35/250; iterations10/ls_iterations20;
physics dt=.005, decimation4. No solver, collision, integrator, asset or motor changes.

## Audit

| Area | Implementation |
|---|---|
| State/reference | Vibed71, native MuJoCo boundary adapters, world/body angular conversion, env origins removed; new packs carry state_layout |
| Observation/goal | obs99, time-major history990; future73×10, f+1 indexing; only synthetic held-pose velocities zeroed |
| Policy | ELU, RunningNorm99/730, FSQ40×9, decoder1030→29; default FSQ true, disabling rejected |
| Critic | 184+730=914, ELU, RunningNorm914; stats updated after actor on H+1 features |
| WM | input133, 2048×3 LayerNorm+SiLU, output70 direct local increments; zero final; no constant-velocity prior |
| Loss/termination | masked dimension-mean/time-sum 8-step loss, out_norm/ERR_FLOOR; analytic hard/soft termination |
| Reward | pelvis anchor; fresh FK from qpos/qvel; f+1 target; shared simulator/imagination math; dt=.02 |
| Noise/bias | collector noise once; execution action used throughout; q_obs=q-q0-bias, q_target=q0+bias+scale*a |
| Replay | time-major full rows, pre-reset s1 snapshots, done masks, ep_step history clipping, per-history xi/f |
| Schedule | warmup64, fixed WM-only500, WM-update iteration counting, UTD.4, LR decay250k |
| DDP | original explicit gradient reduction, sqrt(world-size) LR scaling; global RunningNorm moments |
| Resume | format version, network/norm/optimizer/config/counters/phase, rank-local RNG+SGS files |
| Evaluation | full-clip return denominator, completion, root/body/joint errors and termination causes; manual reset preserves terminal metrics |

## Actual dimensions/indexes (zero-based, end-exclusive)

| Tensor | Fields |
|---|---|
| S71 | p0:3, quat3:7, v_world7:10, w_world10:13, q13:42, qd42:71 |
| obs99 | q-q0-bias0:29, qd29:58, w_body58:61, gravity61:64, a_prev64:93, relative_rot6d93:99 |
| goal73/frame | dp0:3, rot6d3:9, v_body9:12, w_body12:15, q15:44, .1*qd44:73 |
| goal730 | ten future frames in time order |
| input133 | invariant0:71, a_exec71:100, normalized_xi100:133 |
| delta70 | dp0:3, dtheta3:6, dv6:9, dw9:12, dq12:41, dqd41:70 |
| Critic914 | invariant0:71, joint error71:100, body error100:142, root position142:145, rotation145:151, xi151:184, goal184:914 |

## Test evidence

- Full pytest: 42 passed, 17 LAFAN-dependent skipped; one upstream Torch warning.
- Stage targeted pass counts: 13/17/18/20/22/24/26/28/30.
- Real P2 native/model state roundtrip, reference roundtrip, FK body positions and12 physics steps passed.
- Real policy/observation/goal/history reset smoke passed: (2,990)/(2,730).
- Six vectorized steps wrote six complete rows; three timeout s1 values equalled pre-reset snapshots.
- Single-GPU reduced-network run actually executed501 WM updates/1317 env steps and crossed the fixed500 boundary.
- Its checkpoint parameters and buffers were all finite.
- Single-GPU WM-only resume2→4 passed (cumulative env steps69→138).
- Two physical GPUs0/4, NCCL+SGS, WM-only training/resume2→4 passed; cross-rank RunningNorm means/counts checked.
- Two-GPU SGS joint smoke initialized test counter at500, then executed one actor/critic update.
  This is not another actual500-update warmup run. CPU compact-library mode passed.
- Joint two-GPU resume501→503 passed (cumulative env steps67→136).
- G1 synthetic-motion real simulator compatibility passed.
- Pinned Vibed pure-tensor comparison passed: invariant71, targets70, integration71, RunningNorm, goal730.
  Reference FK annotation import is bypassed to avoid Isaac dependency; no cross-robot FK parity is claimed.
- Compiled/eager imagination outputs matched. Small batch compile62.224s;
  eight eager steps.3864s vs compiled.0161s. Not a default-size throughput result.
- Real eval smoke:100-frame stand clip, failed at52 steps; normalized return.296493
  uses denominator99*.1. Full-clip versus actual-step return identity checked.
  Smoke weights do not establish tracking quality.
- Solver and protected physical-file guards passed.
- Old680/930/788 interfaces removed; remaining7:36 slices are explicit native/archive boundaries.
- Python compileall and both shell launcher syntax checks passed.

## Config

Retained robot/P2 assets/joint order/action scale, advanced actuator switch (off),
SGS parameters and sampling modes, CPU/GPU library, DDP, W&B, single/multi-GPU entry
points and resume/checkpoints.
Changed defaults: FSQ true, goal730, obs/history99/990, warmup64; total_iterations
now counts WM rounds; standard launcher batches4096 envs/2048 WM/4096 imagination.
Deprecated/ignored: fixed policy term scales, StateSpec fixed scales/explicit
integration, learned termination settings. Non-default uses warn.
anchor_obs retains an explicitly non-parity9D research input.
Checkpoint format: vibed-parity-state71-v1. Old dev/FSQ checkpoints rejected.
Resume refills replay and preserves phase/optimizer/RNG metadata; it does not
promise bit-identical simulator continuation.

## Reproduction commands

Use the branch environment and P2 asset folder. On docker150 the isolated worktree
uses the source project's interpreter with PYTHONPATH.

~~~bash
export PYTHONPATH=$PWD/src
PY=/root/code/SuperMimic-Reproduced/.venv/bin/python
$PY tests/make_parity_smoke_data.py /tmp/parity-smoke-data
CUDA_VISIBLE_DEVICES=4 $PY -m pytest -q
PARITY_SMOKE_MOTIONS=/tmp/parity-smoke-data/motions CUDA_VISIBLE_DEVICES=4   $PY tests/parity_training_smoke.py
VIBED_REFERENCE_ROOT=/root/code/vibed-parity-reference   $PY tests/parity_reference_smoke.py
PARITY_SMOKE_MOTIONS=/tmp/parity-smoke-data/motions CUDA_VISIBLE_DEVICES=4   $PY tests/parity_compile_smoke.py
PARITY_SMOKE_MOTIONS=/tmp/parity-smoke-data/motions PARITY_SMOKE_UPDATES=2   PARITY_SMOKE_SAMPLER=sgs PARITY_SMOKE_OUTPUT=/tmp/parity-ddp-repro   CUDA_VISIBLE_DEVICES=0,4 torchrun --standalone --nproc-per-node=2   tests/parity_training_smoke.py
# Repeat the last command with PARITY_SMOKE_RESUME=1 for resume.
~~~

Formal single-card command (not run to convergence):

~~~bash
python -m supermimic.scripts.train --env.robot p2 --policy.fsq   --data.library /path/to/p2-library.pt --data.library-device cpu   --env.num-envs 4096 --env.sampling-mode sgs   --schedule.total-iterations 320000 --run-name p2_vibed_parity --wandb.enabled
~~~

For multi-GPU use torchrun --standalone --nproc-per-node=2 before the module and
same arguments. The existing rank4 launcher uses distinct return-sampling shards;
SGS requires the same complete library/order on every rank, so do not combine
that sharded input with SGS.

~~~bash
python scripts/eval_benchmark.py --ckpt /path/to/new/ckpt_latest.pt   --npz-dir /path/to/p2/validation --out /tmp/parity-benchmark.json
~~~

## Intentional differences and unperformed checks

Differences: P2 assets/motors/FK, FSQ40×9 versus continuous32, unchanged mjlab
Contact Solver, original DDP/LR scaling and optional SGS, CPU/GPU library
engineering, and mjlab simulator implementation with shared mathematical heads.

Not performed: converged equal-transition old/new A/B; full P2 corpus benchmark;
default4096-env/2048-batch memory certification; full-size compiled backward
profiling;17 skipped LAFAN tests. These are not marked passed.

For equal-transition A/B, source iterations count env steps and this branch
counts WM rounds. Select a shared actual simulator transition budget, log
env_step_count*num_envs*world_size, and evaluate both with a full-clip denominator.
Old evaluation records can be reaggregated as
old_actual_step_norm_return*executed_steps/(clip_len-1).
Do not directly resume old weights in this branch.

## Modified files

- docs/VIBED_PARITY_BASELINE_AUDIT.md
- docs/VIBED_PARITY_PROGRESS.md
- docs/VIBED_PARITY_REPORT.md
- docs/VIBED_PARITY_SPEC.md
- scripts/bench_imagination.py
- scripts/compare_parity_benchmarks.py
- scripts/data/build_library.py
- scripts/eval_benchmark.py
- scripts/train_h200_baseline_rank2.sh
- scripts/train_h200_rank4.sh
- src/supermimic/config.py
- src/supermimic/data/recorder.py
- src/supermimic/data/replay_buffer.py
- src/supermimic/envs/actions.py
- src/supermimic/envs/mdp/motion_library.py
- src/supermimic/envs/mdp/observations.py
- src/supermimic/envs/mdp/rewards.py
- src/supermimic/envs/mdp/terminations.py
- src/supermimic/envs/state_io.py
- src/supermimic/envs/tracking_env_cfg.py
- src/supermimic/eval/benchmark.py
- src/supermimic/imagination/compiled.py
- src/supermimic/imagination/policy_trainer.py
- src/supermimic/imagination/rollout.py
- src/supermimic/policy/nets.py
- src/supermimic/policy/normalization.py
- src/supermimic/training/loop.py
- src/supermimic/world_model/dynamics.py
- src/supermimic/world_model/heads.py
- src/supermimic/world_model/state_spec.py
- src/supermimic/world_model/wm_trainer.py
- src/supermimic/world_model/world_model.py
- tests/make_parity_smoke_data.py
- tests/parity_compile_smoke.py
- tests/parity_reference_smoke.py
- tests/parity_training_smoke.py
- tests/sgs_smoke.py
- tests/test_analytic_heads.py
- tests/test_env_smoke.py
- tests/test_vibed_action.py
- tests/test_vibed_dynamics.py
- tests/test_vibed_eval.py
- tests/test_vibed_goal_obs.py
- tests/test_vibed_networks.py
- tests/test_vibed_parity_contract.py
- tests/test_vibed_replay.py
- tests/test_vibed_reward.py
- tests/test_vibed_schedule.py
- tests/test_vibed_state_io.py
- tests/test_world_model.py
- tests/vibed_parity_baseline.json

## Equal-budget A/B command

Generate baseline JSON with the source branch's own evaluator and candidate JSON
with this branch's evaluator. Then run:

~~~bash
python scripts/compare_parity_benchmarks.py --baseline-json old.json \
  --candidate-json new.json --baseline-sim-transitions 1000000 \
  --candidate-sim-transitions 1000000 --out comparison.json
~~~

The helper rejects unequal transition budgets and mismatched clip sets/lengths,
and converts only the original source's actual-step-normalized baseline return.
The million-transition values are examples, not measured runs.

Actual final dimension print: state71, obs99, goal730, wm_input133, delta70, critic914. Training termination causes are logged as rank0 counters.
