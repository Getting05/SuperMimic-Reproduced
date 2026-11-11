"""SGS kernel, concurrent ring updates, reset bookkeeping and legacy regressions."""

import io
from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand
from supermimic.envs.mdp.sgs_sampler import SGSSampler


def sampler(n=5, **kwargs):
    return SGSSampler(n, torch.ones(n, dtype=torch.bool), **kwargs)


def test_initial_distribution_and_kernel():
    s = sampler()
    torch.testing.assert_close(s.probabilities(), torch.full((5,), .2))
    for b, rate in enumerate((0, .3, .66, .9, 1)):
        s.update(torch.full((100,), b), torch.arange(100) < round(rate * 100))
    p = s.probabilities()
    assert p[2] > p[0] and p[2] > p[4] and (p > 0).all()
    rates = s.success_rate.double()
    w = (rates + 1e-8).pow(5 * .66) * (1 - rates + 1e-8).pow(5 * .34)
    expected = ((w.clamp_min(1e-8) + 1e-8).log() / 2).softmax(0).float()
    torch.testing.assert_close(p, expected)
    torch.manual_seed(5)
    frequencies = s.sample(200000).bincount(minlength=5) / 200000
    torch.testing.assert_close(frequencies, p, atol=.004, rtol=0)
    assert not torch.allclose(p, torch.full_like(p, .2), atol=.02)
    print("SGS theoretical", p.tolist(), "empirical", frequencies.tolist())


def test_duplicate_bins_ring_overflow_and_restore():
    s = sampler(2)
    # Interleave 137 updates of bin 0 and 123 of bin 1 in a single reset batch.
    bins = torch.cat((torch.zeros(137), torch.ones(123))).long()
    outcomes = torch.arange(260) % 3 == 0
    torch.manual_seed(4)
    order = torch.randperm(260)
    bins, outcomes = bins[order], outcomes[order]
    s.update(bins, outcomes)
    s.update(torch.tensor([0, 0, 1]), torch.tensor([True, False, True]))
    for b in range(2):
        tail = torch.cat((outcomes[bins == b], torch.tensor([True, False]) if b == 0 else torch.tensor([True])))[-100:]
        chronological = s.history[b].roll(-int(s.ptr[b]))
        assert torch.equal(tail, chronological)
        assert s.count[b] == 100
        torch.testing.assert_close(s.success_rate[b], tail.float().mean())
    stream = io.BytesIO()
    torch.save(s.state_dict(), stream)
    stream.seek(0)
    other = sampler(2)
    other.load_state_dict(torch.load(stream, weights_only=True))
    for key, value in s.state_dict().items():
        assert torch.equal(value, other.state_dict()[key])
    torch.testing.assert_close(s.probabilities(), other.probabilities())


def command_stub(lengths=(120, 40), n=3):
    c = MultiClipMotionCommand.__new__(MultiClipMotionCommand)
    # Properties on CommandTerm resolve these from _env.
    c._env = SimpleNamespace(device="cpu", num_envs=n, common_step_counter=0)
    c.cfg = SimpleNamespace(sampling_mode="sgs", sgs_eval_steps=50,
                            sgs_root_pos_threshold=.15, sgs_body_pos_threshold=.15)
    starts = torch.tensor([0, lengths[0]])
    last = starts + torch.tensor(lengths) - 1
    c.library = SimpleNamespace(num_bins=4, bin_start=torch.tensor([0, 50, 100, 120]),
                                bin_end=torch.tensor([50, 100, 120, 160]),
                                bin_clip=torch.tensor([0, 0, 0, 1]), clip_last=last)
    c.sgs_bin_end = torch.minimum(c.library.bin_end, last[c.library.bin_clip] - 50 + 2)
    valid = c.sgs_bin_end > c.library.bin_start
    c.sgs_sampler = SGSSampler(4, valid)
    c.sgs_active = torch.ones(n, dtype=torch.bool)
    c.episode_start_bin = torch.zeros(n, dtype=torch.long)
    c.sgs_steps = torch.zeros(n, dtype=torch.long)
    c.sgs_root_sum = torch.zeros(n)
    c.sgs_body_sum = torch.zeros(n)
    c.sgs_last_step = -1
    c.sgs_finished = torch.zeros(4)
    c._env.termination_manager = SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool),
                                               time_outs=torch.ones(n, dtype=torch.bool))
    return c


def test_short_clip_and_tail_bins_excluded():
    c = command_stub()
    assert c.sgs_sampler.valid.tolist() == [True, True, False, False]
    frames = c._sample_frames(20000)
    assert (frames >= 0).all() and (frames + 49 <= 119).all()
    assert not ((frames >= 100) | (frames >= 120)).any()


def test_success_single_write_failure_and_thresholds():
    c = command_stub()
    c.sgs_steps[:] = 50
    c.sgs_root_sum[:] = torch.tensor([1., 1., 20.])
    c.sgs_body_sum[:] = 1.
    c._env.termination_manager.terminated[1] = True
    ids = torch.arange(3)
    c._finalize_sgs(ids)
    assert c.sgs_sampler.count[0] == 3
    assert c.sgs_sampler.history[0, :3].tolist() == [True, False, False]
    c._finalize_sgs(ids)
    assert c.sgs_sampler.total_updates == 3


def test_early_timeout_is_failure_manual_reset_ignored():
    c = command_stub()
    c.sgs_steps[:] = 49
    c._env.termination_manager.time_outs[2] = False
    c._finalize_sgs(torch.arange(3))
    assert c.sgs_sampler.total_updates == 2 and c.sgs_sampler.success_rate[0] == 0


def test_legacy_start_uniform_and_return_regression():
    c = command_stub()
    lib = c.library
    lib.num_clips, lib.num_frames = 2, 160
    lib.clip_start = torch.tensor([0, 120])
    lib.frame_clip = torch.cat((torch.zeros(120), torch.ones(40))).long()
    lib.last_frame_of = lambda f: lib.clip_last[lib.frame_clip[f]]
    c.cfg.sampling_mode = "start"
    torch.manual_seed(42)
    expected = lib.clip_start[torch.randint(0, 2, (100,))]
    torch.manual_seed(42)
    assert torch.equal(c._sample_frames(100), expected)
    c.cfg.sampling_mode = "uniform"
    torch.manual_seed(42)
    f = torch.randint(0, 160, (100,))
    expected = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
    torch.manual_seed(42)
    assert torch.equal(c._sample_frames(100), expected)
    c.cfg = SimpleNamespace(sampling_mode="return", softmax_temp=25., g_max=10.,
                            memory_alpha=.01, uniform_ratio=.5, hard_ratio=.4, memory_ratio=.1)
    c.bin_return = torch.tensor([1., 3., 5., 9.])
    c.bin_memory = torch.full((4,), .25)
    c.metrics = {"sampling_entropy": torch.zeros(3)}
    hard = (-25 * c.bin_return / 10).softmax(0)
    memory = .99 * c.bin_memory + .01 * hard
    p = .5 / 4 + .4 * hard + .1 * memory / memory.sum()
    p = p / p.sum()
    torch.manual_seed(42)
    b = p.multinomial(100, True)
    f = lib.bin_start[b] + (torch.rand(100) * (lib.bin_end[b] - lib.bin_start[b])).long()
    expected = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])
    torch.manual_seed(42)
    assert torch.equal(c._sample_frames(100), expected)


def test_exact_horizon_counts_terminal_step_once():
    c = command_stub()
    c.robot = SimpleNamespace(data=SimpleNamespace(
        body_link_pos_w=torch.zeros(3, 1, 3),
    ))
    c.robot_root_body_index = 0
    c.body_indexes = torch.tensor([0])
    c._env.scene = SimpleNamespace(env_origins=torch.zeros(3, 3))
    c._ref = SimpleNamespace(root_pos_w=torch.zeros(3, 3))
    c._bodies = SimpleNamespace(pos_w=torch.zeros(3, 1, 3))
    for step in range(1, 51):
        c._env.common_step_counter = step
        horizon = c.record_sgs_step()
        assert horizon.all() == (step == 50)
        c.record_sgs_step()  # Duplicate termination query must be idempotent.
        assert (c.sgs_steps == step).all()
    c._finalize_sgs(torch.arange(3))
    assert c.sgs_sampler.total_updates == 3
    assert c.sgs_sampler.success_rate[0] == 1
    c._finalize_sgs(torch.arange(3))
    assert c.sgs_sampler.total_updates == 3


def test_checkpoint_rejects_dataset_and_configuration_changes():
    c = command_stub()
    from dataclasses import make_dataclass
    Cfg = make_dataclass("Cfg", [("sgs_eval_steps", int, 50)])
    c.cfg = Cfg()
    c.sgs_signature = "dataset-a"
    state = c.sgs_state_dict()
    c.load_sgs_state_dict(state)
    c.sgs_signature = "dataset-b"
    with pytest.raises(ValueError, match="dataset/order"):
        c.load_sgs_state_dict(state)
    c.sgs_signature = "dataset-a"
    c.cfg.sgs_eval_steps = 51
    with pytest.raises(ValueError, match="configuration mismatch"):
        c.load_sgs_state_dict(state)


def test_invalid_configuration_and_excluded_probability():
    with pytest.raises(ValueError):
        SGSSampler(2, torch.zeros(2, dtype=torch.bool))
    s = SGSSampler(3, torch.tensor([True, False, True]))
    assert s.probabilities()[1] == 0
    assert (s.probabilities()[s.valid] > 0).all()
    with pytest.raises(ValueError):
        s.update(torch.tensor([1]), torch.tensor([True]))
