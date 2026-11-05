"""Standalone evaluation controller and isolated CPU/GPU workers.

The controller imports no torch or simulator modules. GPU visibility and CPU
affinity are established in a fresh interpreter before those modules load.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from supermimic.config_io import load_config, update_config
from supermimic.eval.config import METRICS, EvalConfig


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def resource_slots(cfg, allowed=None):
    r = cfg.resources
    if allowed is None:
        saved = os.environ.get("SM_EVAL_ALLOWED_CPU_IDS")
        allowed = [int(i) for i in saved.split(",")] if saved else sorted(os.sched_getaffinity(0))
    allowed = sorted(allowed)
    if not allowed:
        raise ValueError("No available CPUs")
    if r.cpu_ids:
        if not set(r.cpu_ids) <= set(allowed):
            raise ValueError("cpu_ids exceed the launcher's allowed CPUs")
        cores = sorted(r.cpu_ids)
    else:
        capacity = len(allowed)
        quota_file = Path("/sys/fs/cgroup/cpu.max")
        if quota_file.exists():
            quota, period = quota_file.read_text().split()
            if quota != "max":
                capacity = min(capacity, max(1, int(quota) // int(period)))
        count = r.cpu_count if r.cpu_count is not None else (
            max(1, math.floor(capacity * r.cpu_fraction)) if r.cpu_fraction is not None else capacity
        )
        if count > len(allowed):
            raise ValueError("cpu_count exceeds available CPUs")
        # Use the high end by default, keeping the low cores free for training.
        cores = allowed[-count:]
    workers = r.workers
    if r.device == "cuda":
        workers = max(workers, len(r.gpu_ids))
        if workers % len(r.gpu_ids):
            raise ValueError("GPU workers must be a multiple of the GPU count")
    if workers * r.threads_per_worker > len(cores):
        raise ValueError("CPU budget must cover workers * threads_per_worker")
    # Disjoint affinity partitions; thread count is a separate per-worker limit.
    return [cores[i::workers] for i in range(workers)]


def summarize(rows):
    n = len(rows)
    horizon = sum(r["reference_steps"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    result = dict(clips=n, failed=sum(r["failed"] for r in rows),
                  success_rate=sum(r["success"] for r in rows) / max(n, 1),
                  mean_completion_rate=sum(r["completion_rate"] for r in rows) / max(n, 1),
                  mean_norm_return=sum(r["norm_return"] for r in rows) / max(n, 1),
                  reference_frame_weighted_norm_return=sum(r["norm_return"] * r["reference_steps"] for r in rows) / max(horizon, 1),
                  executed_frame_weighted_norm_return=sum(r["executed_norm_return"] * r["steps"] for r in rows) / max(steps, 1),
                  low_quality=sum(r["low_quality"] for r in rows),
                  exclude_candidates=sum(r["exclude_candidate"] for r in rows))
    if rows:
        result["metrics_clip_mean"] = {k: sum(r["metrics"][k] for r in rows) / n for k in rows[0]["metrics"]}
        result["metrics_frame_weighted"] = {k: sum(r["metrics"][k] * r["steps"] for r in rows) / max(steps, 1) for k in rows[0]["metrics"]}
        result["reward_terms_frame_weighted"] = {k: sum(r["terms"][k] * r["steps"] for r in rows) / max(steps, 1) for k in rows[0]["terms"]}
    return result


def export_results(cfg, rows):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation clip IDs")
    rows.sort(key=lambda r: (r["dataset"], r["clip_index"]))
    result = dict(checkpoint=str(cfg.checkpoint.resolve()), config=asdict(cfg),
                  protocol="deterministic-clip-start-v1", metric_units=METRICS,
                  summary=summarize(rows),
                  datasets={d.name: summarize([r for r in rows if r["dataset"] == d.name]) for d in cfg.datasets},
                  clips=rows)
    out = cfg.output_dir
    write_json(out / "results.json", result)
    for name, subset in [("failures", [r for r in rows if r["failed"]]),
                         ("low_quality", [r for r in rows if r["low_quality"]]),
                         ("exclude_candidates", [r for r in rows if r["exclude_candidate"]])]:
        write_json(out / f"{name}.json", subset)
        (out / f"{name}.txt").write_text("".join(r["id"] + "\n" for r in subset))
    flat = [{**{k: v for k, v in r.items() if k not in ("metrics", "terms")},
             **r["metrics"], **{"reward/" + k: v for k, v in r["terms"].items()}} for r in rows]
    with (out / "clips.csv").open("w", newline="") as stream:
        if flat:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return result


def run(cfg):
    cfg.validate()
    if not cfg.checkpoint.is_file():
        raise FileNotFoundError(cfg.checkpoint)
    for dataset in cfg.datasets:
        if not dataset.path.exists():
            raise FileNotFoundError(dataset.path)
    slots = resource_slots(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir = cfg.output_dir / ("workers_" + uuid.uuid4().hex[:12])
    jobs_dir.mkdir()
    jobs = []
    try:
        for rank, cores in enumerate(slots):
            job = jobs_dir / f"{rank:04d}.json"
            output = jobs_dir / f"{rank:04d}.results.json"
            write_json(job, dict(config=asdict(cfg), rank=rank, world=len(slots),
                                 cpu_ids=cores, output=str(output)))
            env = dict(os.environ)
            # A training child must never join the parent's torchrun group.
            for key in list(env):
                if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"} or key.startswith("TORCHELASTIC_"):
                    del env[key]
            env["CUDA_VISIBLE_DEVICES"] = "" if cfg.resources.device == "cpu" else str(cfg.resources.gpu_ids[rank % len(cfg.resources.gpu_ids)])
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(cfg.resources.threads_per_worker)
            with (jobs_dir / f"{rank:04d}.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-m", "supermimic.eval.runner", "--worker", str(job)], env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((process, output))
        while any(p.poll() is None for p, _ in jobs):
            failed = [p.returncode for p, _ in jobs if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker failed: {failed}; see {jobs_dir}")
            progress = []
            for _, output in jobs:
                path = output.with_suffix(".progress.json")
                if path.exists():
                    progress.append(json.loads(path.read_text()))
            write_json(cfg.output_dir / "status.json", dict(state="running", workers=len(jobs), completed=sum(p.poll() == 0 for p, _ in jobs), progress=progress))
            time.sleep(0.2)
        if any(p.returncode != 0 for p, _ in jobs):
            raise RuntimeError(f"Evaluation failed; see {jobs_dir}")
        rows = [row for _, output in jobs for row in json.loads(output.read_text())]
        if not rows:
            raise ValueError("No clips selected")
        result = export_results(cfg, rows)
        write_json(cfg.output_dir / "status.json", dict(state="completed", summary=result["summary"]))
        print(json.dumps(result["summary"]), flush=True)
        return result
    except BaseException as error:
        write_json(cfg.output_dir / "status.json", dict(state="failed", error=str(error), worker_logs=str(jobs_dir)))
        raise
    finally:
        for process, _ in jobs:
            if process.poll() is None:
                process.terminate()
        for process, _ in jobs:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def classify_end(terminated, timeout, clip_end, steps, frames, causes):
    # The command advances before clip_end checks frame+1; natural clip end can
    # occur at L-2 transitions. Physical failure always takes precedence.
    natural = clip_end or steps >= max(frames - 2, 1)
    failed = terminated or (timeout and not clip_end) or not natural
    reason = "+".join(causes) if causes else ("hard_termination" if terminated else "time_out" if failed else "clip_end")
    return failed, reason


def evaluate_dataset(cfg, dataset, rank, world, output):
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint, state_metrics
    from supermimic.config_io import read_config

    device = "cpu" if cfg.resources.device == "cpu" else "cuda:0"
    policy, train_cfg = load_policy_from_checkpoint(cfg.checkpoint, device)
    if cfg.checkpoint_config is not None:
        supplied = read_config(cfg.checkpoint_config)
        # An explicit config is an assertion about the saved model, not a way to
        # silently change its observation layout or actuator/future-frame setup.
        for section in ("policy", "env"):
            for key, value in supplied.get(section, {}).items():
                expected = train_cfg.get(section, {}).get(key)
                if section == "policy" and key == "policy_obs_ref_rot6d":
                    expected = policy.cfg.policy_obs_ref_rot6d
                if json.dumps(value) != json.dumps(expected):
                    raise ValueError(f"checkpoint_config mismatch: {section}.{key}")
    packed = dataset.format == "packed" or (dataset.format == "auto" and dataset.path.suffix == ".pt")
    names = list(dataset.clips)
    if dataset.clip_list is not None:
        names.extend(line.strip() for line in dataset.clip_list.read_text().splitlines() if line.strip())
    files = [str(dataset.path)] if packed else [str(p) for p in discover_npz(dataset.path, names=names or None, manifest=dataset.manifest)]
    if not files:
        raise ValueError(f"No motion files: {dataset.path}")
    if packed and dataset.manifest is not None:
        raise ValueError("Use clips/clip_list for packed libraries; manifest applies to NPZ")
    if packed:
        meta = torch.load(dataset.path, map_location="cpu", weights_only=False)
        candidates = [i for i, name in enumerate(meta["names"]) if not names or name in names]
        del meta
    else:
        candidates = list(range(len(files)))
    if dataset.limit is not None:
        candidates = candidates[:dataset.limit]
    if not candidates:
        raise ValueError(f"Empty dataset selection: {dataset.name}")
    selected = candidates[rank::world]
    if not selected:
        return []
    # NPZ workers only load their own files. Packed storage stays in its native
    # representation; clip indices select each worker's disjoint queue.
    source_indices = selected
    if not packed:
        files = [files[i] for i in selected]
        selected = list(range(len(files)))
    n = min(cfg.resources.num_envs, len(selected))
    ec = make_env_cfg(SuperMimicEnvCfgArgs(
        npz_files=files, compact_library=packed, library_device=cfg.resources.library_device,
        num_envs=n, robot=train_cfg["env"].get("robot", "g1"),
        p2_advanced_actuators=train_cfg["env"].get("p2_advanced_actuators", False),
        future_frames=train_cfg["env"]["future_frames"], anchor_obs=train_cfg["env"].get("anchor_obs", False),
        policy_obs_ref_rot6d=policy.cfg.policy_obs_ref_rot6d,
        episode_length_s=1e5, action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", record_transitions=False, seed=cfg.seed))
    ec.auto_reset = False
    for actuator in ec.scene.entities["robot"].articulation.actuators:
        if hasattr(actuator, "delay_min_lag"):
            actuator.delay_min_lag = actuator.delay_max_lag = cfg.actuator_delay
    env = ManagerBasedRlEnv(ec, device=device)
    rows = []
    wanted = list(dict.fromkeys(cfg.metrics + (list(cfg.quality.max_errors) if cfg.quality.enabled else [])))
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        current = torch.tensor(selected[:n], device=device)
        queue_positions = list(range(n))
        cursor = n
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        active = torch.ones(n, dtype=torch.bool, device=device)
        counts = torch.zeros(n, dtype=torch.long, device=device)
        returns = torch.zeros(n, device=device)
        sums = torch.zeros(n, len(wanted), device=device)
        term_names = list(env.reward_manager.active_terms) if cfg.reward_terms else []
        terms = torch.zeros(n, len(term_names), device=device)
        max_reward = TrackingRewardCfg(dt=env.step_dt).max_step_reward
        ticks = 0
        with torch.inference_mode():
            while bool(active.any()):
                obs, reward, terminated, timeout, _ = env.step(policy(obs["policy"], obs["goal"]))
                if not bool(torch.isfinite(StateIO(env).read()[active]).all()):
                    raise ValueError("Non-finite simulator state")
                counts += active.long()
                returns += reward * active
                values = state_metrics(env, wanted)
                if wanted:
                    sums += torch.stack([values[k] for k in wanted], -1) * active[:, None]
                if term_names:
                    terms += env.reward_manager.term_rewards * active[:, None]
                lengths = lib.clip_len[current]
                done = active & (terminated | timeout | (counts >= lengths - 1))
                for e in done.nonzero().flatten().tolist():
                    st, frames = int(counts[e]), int(lengths[e])
                    tm = env.termination_manager
                    clip_end = bool(tm.get_term("clip_end")[e])
                    causes = [key for key in ("root_height", "root_tilt") if bool(tm.get_term(key)[e])]
                    failed, reason = classify_end(bool(terminated[e]), bool(timeout[e]), clip_end, st, frames, causes)
                    completion = min(st / max(frames - 2, 1), 1.0)
                    metrics = {key: float(sums[e, j]) / max(st, 1) for j, key in enumerate(wanted)}
                    flags = []
                    if cfg.quality.enabled:
                        if completion < cfg.quality.min_completion:
                            flags.append("low_completion")
                        flags.extend("high_" + key for key, bound in cfg.quality.max_errors.items() if metrics[key] > bound)
                    original_index = source_indices[queue_positions[e]]
                    name = lib.clips[int(current[e])].name
                    rows.append(dict(id=f"{dataset.name}:{original_index}:{name}", dataset=dataset.name,
                                     clip_index=original_index, name=name, frames=frames, reference_steps=max(frames - 1, 1),
                                     steps=st, failed=failed, success=not failed, termination_reason=reason,
                                     completion_rate=completion, norm_return=float(returns[e]) / (max_reward * max(frames - 1, 1)),
                                     executed_norm_return=float(returns[e]) / (max_reward * max(st, 1)),
                                     metrics=metrics, terms={key: float(terms[e, j]) / max(st, 1) for j, key in enumerate(term_names)},
                                     low_quality=bool(flags), quality_flags=flags, exclude_candidate=failed or bool(flags)))
                    if cursor < len(selected):
                        current[e] = selected[cursor]
                        queue_positions[e] = cursor
                        cursor += 1
                    else:
                        active[e] = False
                    counts[e] = 0
                    returns[e] = 0
                    sums[e] = 0
                    terms[e] = 0
                reset = (done | terminated | timeout).nonzero().flatten()
                if reset.numel() and bool(active.any()):
                    cmd.forced_frame[reset] = lib.clip_start[current[reset]]
                    obs, _ = env.reset(env_ids=reset)
                ticks += 1
                if ticks % 100 == 0 or bool(done.any()):
                    write_json(output.with_suffix(".progress.json"), dict(dataset=dataset.name, completed=len(rows), total=len(selected)))
        return rows
    finally:
        env.close()


def worker(path):
    job = json.loads(Path(path).read_text())
    os.sched_setaffinity(0, job["cpu_ids"])
    cfg = update_config(EvalConfig(), job["config"])
    cfg.validate()
    import torch
    torch.set_num_threads(cfg.resources.threads_per_worker)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg.seed)
    output = Path(job["output"])
    rows = []
    for dataset in cfg.datasets:
        rows.extend(evaluate_dataset(cfg, dataset, job["rank"], job["world"], output))
    write_json(output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override config checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list-metrics", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list_metrics:
        print(json.dumps(METRICS, indent=2))
    elif args.worker:
        worker(args.worker)
    else:
        if args.config is None:
            parser.error("--config is required")
        cfg = load_config(args.config, EvalConfig())
        if args.checkpoint:
            cfg.checkpoint = args.checkpoint
        if args.output_dir:
            cfg.output_dir = args.output_dir
        run(cfg)


if __name__ == "__main__":
    main()
