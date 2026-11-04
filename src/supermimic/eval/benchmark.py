"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.policy.nets import Policy, PolicyCfg


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    metrics: dict | None = None
    completion_rate: float = 0.0
    executed_norm_return: float = 0.0
    termination_reason: str = ""
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format_version") != "vibed-parity-state71-v1":
        raise ValueError("Old or incompatible checkpoint: explicit Vibed parity checkpoint required")
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, robot: str = "g1", p2_advanced_actuators: bool = False) -> list[ClipResult]:
    num_envs = min(num_envs or 1024, len(npz_files))
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        robot=robot,
        p2_advanced_actuators=p2_advanced_actuators,
    )
    env = make_env(args, device=device)
    env.cfg.auto_reset=False
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = 0.1  # paper reward upper bound per step (dt * 5)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        metric_sum = None
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"])
            obs, rew, term, tout, _ = env.step(act)
            sample_metrics = state_metrics(env)
            if metric_sum is None: metric_sum=torch.zeros(num_envs,len(sample_metrics),device=device)
            metric_sum += torch.stack(list(sample_metrics.values()),-1)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = term | tout
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(lib.clips[ci].length-1, 1)),
                            executed_norm_return=float(ret_sum[e])/(max_r*max(st,1)),
                            completion_rate=st/max(lib.clips[ci].length-1,1),
                            metrics={name:float(metric_sum[e,j])/max(st,1) for j,name in enumerate(sample_metrics)},
                            termination_reason=termination_reason(env,e) if bool(term[e]) else "clip_end",
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(current[ids],max=n_clips-1)]
                obs,_=env.reset(env_ids=ids)
                metric_sum[ids]=0.0
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def termination_reason(env,e):
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.mdp.rewards import simulator_heads
    from supermimic.world_model.wm_trainer import reference_at
    s=StateIO(env).read()
    cmd=env.command_manager.get_term("motion")
    err=simulator_heads(env).termination_features(s[:,:3],s[:,3:7],reference_at(cmd.library,cmd.frame)).abs()
    causes=[]
    if err[e,0]>.25:causes.append("pelvis_height")
    if err[e,1]>.8:causes.append("pelvis_tilt")
    return "+".join(causes) or "hard_termination"


def state_metrics(env):
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.mdp.rewards import simulator_heads,reanchor_reference
    from supermimic.world_model.wm_trainer import reference_at
    from supermimic.kinematics.rotations import quat_error_magnitude
    s=StateIO(env).read()
    cmd=env.command_manager.get_term("motion")
    ref=reference_at(cmd.library,cmd.frame)
    r=cmd.library.ref(cmd.frame)
    b=simulator_heads(env).robot_bodies(s)
    rp,rq=reanchor_reference(ref.body_pos_w,ref.body_quat_w,ref.root_pos_w,
        ref.root_quat_w,s[:,:3],s[:,3:7])
    return dict(pelvis_position=(s[:,:3]-ref.root_pos_w).norm(dim=-1),
        pelvis_orientation=quat_error_magnitude(s[:,3:7],ref.root_quat_w),
        body_position_world=(b["pos"]-ref.body_pos_w).norm(dim=-1).mean(-1),
        body_orientation_world=quat_error_magnitude(b["quat"],ref.body_quat_w).mean(-1),
        body_position_reanchored=(b["pos"]-rp).norm(dim=-1).mean(-1),
        body_orientation_reanchored=quat_error_magnitude(b["quat"],rq).mean(-1),
        joint_position=(s[:,13:42]-r.joint_pos).abs().mean(-1),
        joint_velocity=(s[:,42:71]-r.joint_vel).abs().mean(-1))


def summarize(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * max(r.frames-1,1) for r in results) / max(sum(max(r.frames-1,1) for r in results),1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    out["reference_frame_weighted_norm_return"]=w_ret
    out["mean_completion_rate"]=sum(r.completion_rate for r in results)/max(n,1)
    if results and results[0].metrics:
        out["metrics_clip_mean"]={k:sum(r.metrics[k] for r in results)/n for k in results[0].metrics}
        out["metrics_frame_weighted"]={k:sum(r.metrics[k]*r.steps for r in results)/max(tot_steps,1) for k in results[0].metrics}
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
