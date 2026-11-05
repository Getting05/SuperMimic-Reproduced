from dataclasses import asdict
import pytest
import torch
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.eval.benchmark import load_policy_from_checkpoint

@pytest.mark.parametrize("ref6d", [False, True])
def test_observation_modes_and_legacy_checkpoint(tmp_path, ref6d):
    cfg = PolicyCfg(policy_obs_ref_rot6d=ref6d, encoder_hidden=(16,), decoder_hidden=(16,))
    policy = Policy(cfg)
    assert policy(torch.zeros(2, 10, 99 if ref6d else 93), torch.zeros(2, 730)).shape == (2, 29)
    pc = asdict(cfg)
    del pc["policy_obs_ref_rot6d"]
    path = tmp_path / "legacy.pt"
    torch.save(dict(format_version="vibed-parity-state71-v1", policy=policy.state_dict(),
                    config={"policy": pc}), path)
    restored, _ = load_policy_from_checkpoint(path, "cpu")
    assert restored.cfg.policy_obs_ref_rot6d == ref6d

def test_default_excludes_reference_orientation():
    assert not PolicyCfg().policy_obs_ref_rot6d
