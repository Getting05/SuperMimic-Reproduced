# Vibed parity implementation status

Status: stage 2 state boundaries implemented and tested. Stages 3–10 remain open.

Source: `0ad10dfb043077fb9c7536169503e74a60a960e4` (origin/dev/FSQ).
Reference: `0525a23a2dcfa61f8543f98b84c075fe66d9d286` (Vibed main).
Branch: `dev/vibed-parity-fsq`.
Worktree: `/root/code/SuperMimic-vibed-parity`.
The source worktree was clean and has not been modified.

## Solver decision (resolved by user)

The attachment requires both preserving all Contact Solver settings and explicitly
specifies nconmax=35/njmax=250. The actual latest source selects nconmax=128/njmax=768
for P2, and 35/250 for G1. Both have iterations=10 and ls_iterations=20.
No solver setting has been changed. The baseline test protects the actual source
configuration, per the user's instruction to preserve the current solver. Do not silently revert P2 to 35/250.

## Completed validation

- Baseline protection + dimension arithmetic + SGS suite: 13 passed in 8.88s.
- Baseline dimensional/slice scan recorded 50 candidate lines for later review.

- Existing SGS unit tests: 9 passed (one upstream Torch deprecation warning).
- P2 FSQ CUDA smoke on physical GPU 4: 52 vectorized env steps, 2 envs, 128 replay
  capacity, WM and policy updates executed, BASELINE_TRAIN_OK, exit 0.
- Smoke used the existing stand NPZ at /tmp/supermimic-p2-smoke.Muge4b/motions.
  This is only a synthetic pipeline check, not a tracking quality benchmark.
- Smoke options: warmup=12, UTD=.05, WM batches=1/batch=8, policy batch=8/horizon=8,
  critic epochs=1/minibatches=1, compile disabled.
- Baseline log reported nse1=.0783 and nse8=nan. This smoke does NOT establish
  finite 8-step WM evaluation; the nan requires diagnosis before claiming that
  acceptance criterion passes.
- An earlier smoke invocation used a nonexistent compact-library path and failed
  with FileNotFoundError. The subsequent successful run used the existing NPZ.
- Single-card parity, multi-card parity, resume, eval, reward equivalence and
  all new tensor-layout acceptance tests remain unperformed.

## Environment

The worktree reuses the source installation through PYTHONPATH and a local
untracked asset symlink. No environment or source install was changed.
The asset symlink must not be committed.

## Test command

```bash
cd /root/code/SuperMimic-vibed-parity
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD/src MUJOCO_GL=egl \
  /root/code/SuperMimic-Reproduced/.venv/bin/python -m pytest \
  tests/test_vibed_parity_contract.py tests/test_sgs_sampler.py -q
```

## Remaining work

Stages 3–10 in VIBED_PARITY_SPEC.md remain open.
930D history and 680D goal are still active. Old training checkpoints are now rejected before loading weights, using an interim state-layout format version. Full resume protocol remains stage 10. No parity completion is claimed.

## Stage 2 evidence

16 targeted tests passed in 8.74s. A real P2 CUDA integration checked StateIO read/write, reference native/model roundtrip, FK body positions versus simulator, and 12 physics steps: STATE71_SIM_REFERENCE_FK_OK, shape (2,71). Contact Solver and physical asset guards passed. The existing WM integration mathematics remains pending replacement in stage 8.
