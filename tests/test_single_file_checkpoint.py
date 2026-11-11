from copy import deepcopy
from types import SimpleNamespace
import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer


def tiny_trainer(path,rank=0,world=1,bins=3):
    t=Trainer.__new__(Trainer)
    t.rank,t.world,t.is_main,t.device=rank,world,rank==0,'cpu'
    t.cfg=TrainConfig();t.cfg.env.robot='p2';t.log_dir=path
    t.wm=torch.nn.Linear(2,2);t.policy=torch.nn.Linear(2,2);t.critic=torch.nn.Linear(2,2)
    t.wm_trainer=SimpleNamespace(opt=torch.optim.Adam(t.wm.parameters()))
    t.policy_trainer=SimpleNamespace(opt=torch.optim.Adam(t.policy.parameters()),critic_opt=torch.optim.Adam(t.critic.parameters()))
    t.command=SimpleNamespace(sgs_sampler=None,bin_return=torch.arange(bins,dtype=torch.float32),bin_memory=torch.full((bins,),1/bins))
    t.iteration=1000;t.num_updates=1000;t.env_step_count=2564;t._utd_acc=.25
    return t


def test_one_file_roundtrip_includes_sampler_and_rng(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda *a:torch.get_rng_state())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda *a:None)
    t=tiny_trainer(tmp_path);t.save('one.pt')
    assert [p.name for p in tmp_path.iterdir()]==['one.pt']
    ck=torch.load(tmp_path/'one.pt',weights_only=False)
    assert 'rank_files' not in ck and len(ck['rank_states'])==1
    assert ck['rank_states'][0]['bin_return'].device.type=='cpu'
    other=tiny_trainer(tmp_path);other.command.bin_return.zero_();other.load(tmp_path/'one.pt')
    torch.testing.assert_close(other.command.bin_return,torch.arange(3,dtype=torch.float32))
    assert other.iteration==1000 and other.env_step_count==2564
    assert other._utd_acc==.25
    for a,b in zip(t.policy.parameters(),other.policy.parameters()):torch.testing.assert_close(a,b)


def test_one_file_restores_different_rank_shard_sizes(tmp_path,monkeypatch):
    import supermimic.training.loop as loop
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda *a:torch.get_rng_state())
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda *a:None)
    def gather(local,states,dst):
        for rank in range(4):
            states[rank]=deepcopy(local)
            states[rank]['bin_return']=torch.full((rank+3,),float(rank))
            states[rank]['bin_memory']=torch.full((rank+3,),1/(rank+3))
    monkeypatch.setattr(loop.dist,'gather_object',gather)
    root=tiny_trainer(tmp_path,world=4);root.save('four.pt')
    assert len(list(tmp_path.iterdir()))==1
    for rank in range(4):
        other=tiny_trainer(tmp_path,rank=rank,world=4,bins=rank+3)
        other.load(tmp_path/'four.pt')
        torch.testing.assert_close(other.command.bin_return,torch.full((rank+3,),float(rank)))



def test_cuda_map_location_restores_cpu_rng(tmp_path):
    import pytest
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    t=tiny_trainer(tmp_path);t.device="cuda:0";t.save("gpu.pt")
    other=tiny_trainer(tmp_path);other.device="cuda:0";other.load(tmp_path/"gpu.pt")
    assert other.iteration==1000
    torch.testing.assert_close(other.command.bin_return,t.command.bin_return)
