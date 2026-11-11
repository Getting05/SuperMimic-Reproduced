"""Configuration, resource isolation and evaluation protocol regression tests."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from supermimic.config_io import load_config, update_config
from supermimic.eval.config import DatasetCfg, EvalConfig, ResourceCfg
from supermimic.eval.runner import classify_end, export_results, resource_slots


def test_toml_nested_config_and_json_roundtrip(tmp_path):
    p = tmp_path / "eval.toml"
    p.write_text('checkpoint="a.pt"\n[[datasets]]\nname="test"\npath="b.pt"\n[resources]\ndevice="cpu"\ncpu_ids=[2,3]\nworkers=2\n')
    cfg = load_config(p, EvalConfig())
    cfg.validate()
    assert cfg.datasets[0].path == Path("b.pt")
    assert resource_slots(cfg, [0, 1, 2, 3]) == [[2], [3]]
    restored = update_config(EvalConfig(), json.loads(json.dumps(asdict(cfg), default=str)))
    assert restored == cfg
    with pytest.raises(ValueError, match="Unknown"):
        update_config(cfg, {"typo": True})
    with pytest.raises(ValueError):
        update_config(cfg, {"resources": {"workers": "four"}})


def test_gpu_mapping_and_cpu_oversubscription():
    cfg = EvalConfig(resources=ResourceCfg(device="cuda", gpu_ids=[6, 7], cpu_ids=[4, 5, 6, 7]))
    assert resource_slots(cfg, list(range(8))) == [[4, 6], [5, 7]]
    cfg.resources.workers = 3
    with pytest.raises(ValueError, match="multiple"):
        resource_slots(cfg, list(range(8)))
    cfg.resources.device = "cpu"
    cfg.resources.workers = 8
    with pytest.raises(ValueError, match="budget"):
        resource_slots(cfg, list(range(8)))


def test_conflicting_resource_selectors_and_metrics_rejected():
    cfg = EvalConfig(resources=ResourceCfg(cpu_ids=[1], cpu_count=1))
    with pytest.raises(ValueError, match="Choose"):
        cfg.validate()
    cfg = EvalConfig(metrics=["missing"])
    with pytest.raises(ValueError, match="Unknown metrics"):
        cfg.validate()


def test_natural_end_and_physical_failure_are_distinct():
    assert classify_end(False, True, True, 98, 100, []) == (False, "clip_end")
    assert classify_end(True, True, True, 98, 100, ["root_tilt"]) == (True, "root_tilt")
    assert classify_end(False, True, False, 5, 100, [])[0]
    assert classify_end(False, False, False, 99, 100, [])[0] is False


def test_exports_keep_full_horizon_and_surviving_step_returns_separate(tmp_path):
    cfg = EvalConfig(output_dir=tmp_path, datasets=[DatasetCfg(name="test")])
    base = dict(dataset="test", frames=101, reference_steps=100, steps=10,
                completion_rate=.1, failed=True, success=False,
                norm_return=.05, executed_norm_return=.5, low_quality=False,
                exclude_candidate=True, metrics={"joint_position": .2}, terms={})
    rows = [dict(base, id="a", name="a", clip_index=0),
            dict(base, id="b", name="b", clip_index=1, failed=False, success=True,
                 norm_return=1., steps=100, executed_norm_return=1.)]
    result = export_results(cfg, rows)
    assert result["summary"]["reference_frame_weighted_norm_return"] == .525
    assert result["summary"]["success_rate"] == .5
    assert (tmp_path / "failures.txt").read_text() == "a\n"
    assert (tmp_path / "clips.csv").is_file()


def test_joint_metric_uses_vibed71_joint_slice(monkeypatch):
    import torch
    from types import SimpleNamespace
    from supermimic.envs.state_io import StateIO
    from supermimic.eval import benchmark
    import supermimic.world_model.wm_trainer as wm
    state = torch.zeros(1, 71)
    state[:, 7:13] = 999  # linear/angular velocity must never enter joint MAE
    state[:, 13:42] = 2
    state[:, 42:71] = 3
    ref = SimpleNamespace(joint_pos=torch.ones(1, 29), joint_vel=torch.ones(1, 29))
    cmd = SimpleNamespace(frame=torch.zeros(1, dtype=torch.long), library=SimpleNamespace(ref=lambda f: ref))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: cmd))
    monkeypatch.setattr(StateIO, "__init__", lambda self, env: None)
    monkeypatch.setattr(StateIO, "read", lambda self: state)
    monkeypatch.setattr(wm, "reference_at", lambda *args: ref)
    metrics = benchmark.state_metrics(env, ["joint_position", "joint_velocity"])
    assert metrics["joint_position"].item() == 1.
    assert metrics["joint_velocity"].item() == 2.


def test_automatic_eval_backpressure_and_failure_recording(tmp_path, monkeypatch):
    from supermimic.eval.automatic import AutoEvaluator
    from supermimic.eval.config import AutoEvalCfg
    import supermimic.eval.automatic as automatic
    calls = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

    def start(args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr(automatic.subprocess, "Popen", start)
    evaluator = AutoEvaluator(AutoEvalCfg(enabled=True, wait_on_finish=False), tmp_path)
    evaluator.submit(tmp_path / "ckpt_0001000.pt")
    evaluator.submit(tmp_path / "ckpt_0002000.pt")
    assert len(calls) == 1
    assert evaluator.history[-1]["state"] == "skipped_busy"
    resolved = json.loads(Path(calls[0][0][-1]).read_text())
    assert resolved["checkpoint"] == str(tmp_path / "ckpt_0001000.pt")
    assert calls[0][1]["start_new_session"] is True
    evaluator.process.returncode = 1
    evaluator.poll()
    assert evaluator.history[0]["state"] == "failed"
    evaluator.submit(tmp_path / "ckpt_0002000.pt")
    assert len(calls) == 2
    evaluator.process.returncode = 0
    evaluator.poll()
    assert evaluator.history[-1]["state"] == "completed"
