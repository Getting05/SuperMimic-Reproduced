import torch
import pytest
from types import SimpleNamespace
from supermimic.config import ScheduleCfg
from supermimic.training.loop import WM_ONLY_UPDATES
from supermimic.imagination.policy_trainer import sample_valid_starts

def test_schedule_contract():
    c=ScheduleCfg()
    assert c.warmup_iterations==64 and c.total_iterations==320000
    assert WM_ONLY_UPDATES==500 and c.updates_per_env_step==.4
    acc=0.;steps=0
    for _ in range(800000):
        acc+=.4
        if acc>=1:acc-=1;steps+=1
    assert steps==320000

def test_short_clip_fails_without_sampling():
    lib=SimpleNamespace(clip_len=torch.tensor([12]))
    with pytest.raises(ValueError,match="No clip"):
        sample_valid_starts(None,lib,8,10,32)
