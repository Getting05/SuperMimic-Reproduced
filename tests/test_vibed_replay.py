import torch
from supermimic.data.replay_buffer import ReplayBuffer

def fill(buf,t,done=False,ep=0):
    x=torch.full((2,71),float(t))
    buf.add(s=x,s1=x+1,a=torch.full((2,29),float(t)),a_prev=torch.full((2,29),float(t-1)),
        xi=torch.zeros(2,33),r=torch.ones(2),f=torch.full((2,),t,dtype=torch.long),
        ep_step=torch.full((2,),ep,dtype=torch.int32),done=torch.full((2,),done))

def test_time_major_done_and_ring():
    b=ReplayBuffer(2,8,71,29,33,"cpu")
    for t in range(12):fill(b,t,done=t%3==2,ep=t%3)
    assert b.s.shape==(8,2,71) and b.size==16
    w=b.sample_windows(100,4)
    assert w.mask.shape==(4,100)
    assert torch.equal(w.mask[1],~w.done[0])
    assert not (w.mask[2] & w.done[0]).any()
    torch.testing.assert_close(w.states[1:,:,0],w.a_exec[:,:,0]+1)

def test_history_never_crosses_episode():
    b=ReplayBuffer(2,8,71,29,33,"cpu")
    for t in range(8):fill(b,t,done=t==3,ep=t%4)
    s=b.sample_starts(200,10)
    assert torch.equal(s.hist_states[-1],s.state)
    assert ((s.hist_states[:,:,0]//4)==(s.state[:,0]//4)[None]).all()
    assert (s.hist_states[:,:,0]<=s.state[:,0][None]).all()
