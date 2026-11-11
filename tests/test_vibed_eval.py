import torch
import pytest
from supermimic.eval.benchmark import ClipResult,summarize,load_policy_from_checkpoint

def test_full_clip_aggregate_weights_reference_length():
    a=ClipResult("a",101,10,.05,True,10)
    b=ClipResult("b",11,10,1.,False,-1)
    s=summarize([a,b])
    assert s["failed"]==1
    assert abs(s["step_weighted_norm_return"]-15/110)<1e-8

def test_eval_rejects_legacy_checkpoint(tmp_path):
    p=tmp_path/"old.pt";torch.save({},p)
    with pytest.raises(ValueError,match="incompatible"):
        load_policy_from_checkpoint(p,"cpu")

def test_ab_reaggregation_and_equal_budget():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location("compare",Path(__file__).resolve().parents[1]/"scripts/compare_parity_benchmarks.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a={"clips":[{"name":"x","frames":11,"steps":5,"norm_return":1.}]}
    b={"clips":[{"name":"x","frames":11,"steps":5,"norm_return":.4}]}
    assert abs(m.compare(a,b,100,100)["mean_delta"]+.1)<1e-7
    with pytest.raises(ValueError,match="equal"):m.compare(a,b,100,101)
