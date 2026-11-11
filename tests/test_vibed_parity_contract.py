"""Baseline guards; these do not claim that algorithm parity is implemented."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests/vibed_parity_baseline.json").read_text())

def test_protected_physical_assets_unchanged():
    for name, digest in BASELINE["protected_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

@pytest.mark.parametrize("robot,nconmax,njmax", [("p2",128,768),("g1",35,250)])
def test_contact_solver_matches_actual_source_baseline(robot,nconmax,njmax):
    from supermimic.envs import SuperMimicEnvCfgArgs
    from supermimic.envs.tracking_env_cfg import make_env_cfg
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[],robot=robot))
    assert cfg.sim.nconmax == nconmax
    assert cfg.sim.njmax == njmax
    assert cfg.sim.mujoco.iterations == 10
    assert cfg.sim.mujoco.ls_iterations == 20
    assert cfg.sim.mujoco.timestep == .005

def test_spec_dimension_arithmetic():
    assert 3+4+3+3+29+29 == 71
    assert 29+29+3+3+29+6 == 99
    assert (3+6+3+3+29+29)*10 == 730
    assert 71+29+33 == 133
    assert 3+3+3+3+29+29 == 70
    assert 71+29+14*3+3+6+33+730 == 914
    assert 40+10*99 == 1030
