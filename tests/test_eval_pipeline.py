"""Real P2 CPU simulation through two isolated evaluation workers."""
from dataclasses import asdict
import json

import torch

from supermimic.envs.p2 import P2_JOINT_NAMES, load_p2_model
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.eval.config import DatasetCfg, EvalConfig, ResourceCfg
from supermimic.eval.runner import run


def test_two_cpu_workers_cover_packed_clips_once(tmp_path):
    model = load_p2_model()
    ref = torch.zeros(3 * 16, 71)
    ref[:, :3] = torch.tensor(model.qpos0[:3])
    ref[:, 3:7] = torch.tensor(model.qpos0[3:7])
    ref[:, 13:42] = torch.tensor(model.qpos0[7:])
    library = tmp_path / "motions.pt"
    torch.save(dict(names=["stand0", "stand1", "stand2"], starts=[0, 16, 32],
                    lengths=[16, 16, 16], ref=ref, state_layout="vibed71",
                    joint_names=P2_JOINT_NAMES), library)
    pc = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(format_version="vibed-parity-state71-v1", policy=Policy(pc).state_dict(),
                    config=dict(policy=asdict(pc), env=dict(robot="p2", future_frames=10,
                                p2_advanced_actuators=False, anchor_obs=False))), checkpoint)
    cfg = EvalConfig(checkpoint=checkpoint, output_dir=tmp_path / "out",
                     datasets=[DatasetCfg(name="test", path=library)],
                     resources=ResourceCfg(device="cpu", cpu_count=2, workers=2, num_envs=2))
    result = run(cfg)
    assert len(result["clips"]) == 3
    assert {r["name"] for r in result["clips"]} == {"stand0", "stand1", "stand2"}
    assert all(r["steps"] > 0 for r in result["clips"])
    assert all("joint_position" in r["metrics"] for r in result["clips"])
    assert json.loads((cfg.output_dir / "status.json").read_text())["state"] == "completed"
