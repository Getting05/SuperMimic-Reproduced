# Baseline dimensional and slice audit

Source: 0ad10dfb043077fb9c7536169503e74a60a960e4. These are inspection candidates, not confirmed bugs. No algorithm changes have been applied.

## src/supermimic/data/motion_convert.py

```text
10:   cols 7:36  29 joint angles in :data:`G1_JOINT_NAMES` order
```

## src/supermimic/envs/mdp/commands.py

```text
539:     gamma: float = 0.99
```

## src/supermimic/envs/mdp/motion_library.py

```text
12: * **compact** (BONES-SEED, 47.5M frames): per frame only the 71-d reference state
34: REF_DIM = 71
195:         return RefState(r[..., 0:3], r[..., 3:7], r[..., 7:36], r[..., 36:39], r[..., 39:42], r[..., 42:71])
229:         return self._ref[:, 7:36].to(device=self.device, dtype=self.dtype)
233:         return self._ref[:, 36:39].to(device=self.device, dtype=self.dtype)
237:         return self._ref[:, 39:42].to(device=self.device, dtype=self.dtype)
241:         return self._ref[:, 42:71].to(device=self.device, dtype=self.dtype)
```

## src/supermimic/envs/state_io.py

```text
1: """Read / write the full simulator state (qpos | qvel) as one 71-d tensor."""
25:         """(N, 71) copy of [qpos | qvel]."""
```

## src/supermimic/imagination/policy_trainer.py

```text
24:     gamma: float = 0.99
```

## src/supermimic/imagination/rollout.py

```text
17: # Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
24:     states: torch.Tensor  # (H+1, n, 71)
29:     goals: torch.Tensor  # (H+1, n, 680)
```

## src/supermimic/policy/nets.py

```text
13: GOAL_DIM = 680
31:     goal_dim: int = GOAL_DIM  # 680 (+9 with anchor_obs)
71:     def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
94:     """V(canonical state 71, goal 680, xi 33, root error 4) -> scalar (one per interface)."""
96:     def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
```

## src/supermimic/world_model/dynamics.py

```text
34:     def __init__(self, feat_dim: int = 71, act_dim: int = 29, xi_dim: int = 33, hidden=(2048, 2048, 2048), out_dim: int = 70):
```

## src/supermimic/world_model/heads.py

```text
3: Everything here is a pure differentiable function of the 71-d simulator state
20: OBS_DIM = 93
27:     return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]
33:     """Noise-free 93-d proprioceptive observation (paper Table A.1) from state + context.
58:     """(H, N, 93) oldest→newest  ->  (N, 930) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
128:             joint_pos=s_next[..., 7:36],
```

## src/supermimic/world_model/state_spec.py

```text
4:                       root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)
6: Canonical features (network input, heading-invariant, 71):
40: STATE_DIM = 71
41: FEAT_DIM = 71
76:         return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]
79:         """Heading-invariant, scaled network input features (…, 71)."""
94:         """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
```

## src/supermimic/world_model/world_model.py

```text
80:         states: (k+1, n, 71) logged states; a_exec: (k, n, 29); xi: (n, 33);
```

## scripts/bench_imagination.py

```text
46:         hist_states=s0[None].expand(10, batch, 71).contiguous(),
57:             J = policy_objective(ro, None, 0.99)
```

## scripts/data/build_library.py

```text
1: """Pack filtered npz clips into one compact library file (fp16 71-d reference states).
```

## scripts/media/render_rollout.py

```text
45:     <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.91 0.93" rgb2="0.76 0.78 0.82" width="512" height="512" mark="edge" markrgb="0.68 0.70 0.74"/>
```

## tests/test_analytic_heads.py

```text
169:             np.testing.assert_allclose(ours.cpu().numpy(), obs["goal"][:, 680:].cpu().numpy(), atol=3e-5)
```

## tests/test_env_smoke.py

```text
32:     assert obs["policy"].shape == (32, 930)
33:     assert obs["goal"].shape == (32, 680)
66:     assert w.states.shape == (9, 256, 71) and w.a_exec.shape == (8, 256, 29) and w.xi.shape == (256, 33)
72:     assert s.hist_states.shape == (10, 128, 71) and s.hist_actions.shape == (10, 128, 29)
81:     assert s.shape == (32, 71)
```

## tests/test_sgs_sampler.py

```text
141:     memory = .99 * c.bin_memory + .01 * hard
```

## tests/test_world_model.py

```text
14:     s = torch.randn(n, 71, device=device, dtype=dtype)
36:     s_rot[:, 36:39] = quat_apply(qy, s[:, 36:39])
41:     np.testing.assert_allclose(s3[:, 0:3].numpy(), (s[:, 0:3] + 0.02 * s[:, 36:39]).numpy(), atol=1e-12)
42:     np.testing.assert_allclose(s3[:, 7:36].numpy(), (s[:, 7:36] + 0.02 * s[:, 42:71]).numpy(), atol=1e-12)
```
