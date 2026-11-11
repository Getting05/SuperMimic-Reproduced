"""Real-library scale validation without changing training algorithms or phase constants."""
import json,os,time
from pathlib import Path
import supermimic.scripts.train
import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer,execution_action
out=Path(os.environ.get('PARITY_VALIDATION_OUTPUT','logs/validation_20261104/scale'))
if int(os.environ.get('WORLD_SIZE','1'))>1:out=out/f'rank_{os.environ["RANK"]}'
out.mkdir(parents=True,exist_ok=True)
c=TrainConfig();c.env.robot='p2';c.policy.fsq=True
c.env.num_envs=int(os.environ.get('PARITY_VALIDATION_ENVS','4096'))
c.data.library=Path(os.environ['PARITY_VALIDATION_LIBRARY'])
c.data.library_device=os.environ.get('PARITY_VALIDATION_STORAGE','cpu')
c.env.sampling_mode='sgs'
c.log_root=out;c.run_name='run'
# First validate eager arithmetic/memory; compiled behavior is a separate test.
c.wm_trainer.compile=None;c.policy_trainer.compile=None
t=Trainer(c)
result=dict(config=dict(num_envs=c.env.num_envs,wm_batch=c.wm_trainer.batch_size,
  wm_minibatches=c.wm_trainer.batches_per_update,window=c.wm_trainer.window,
  policy_batch=c.policy_trainer.batch_size,horizon=c.policy_trainer.horizon,
  library=str(c.data.library),storage=c.data.library_device,
  wm_hidden=c.world_model.hidden,encoder_hidden=c.policy.encoder_hidden),
  phases={})
def mark(phase,**data):
 result['phases'][phase]=data
 (out/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(phase=phase,**data)),flush=True)
try:
 obs,_=t.env.reset()
 mark('initialized',clips=t.lib.num_clips,frames=t.lib.num_frames,
      replay_gib=sum(x.numel()*x.element_size() for x in
      (t.buffer.s,t.buffer.s1,t.buffer.a,t.buffer.a_prev,t.buffer.xi,t.buffer.r,
       t.buffer.f,t.buffer.ep_step,t.buffer.done))/1024**3)
 start=time.perf_counter()
 for i in range(64):
  a=execution_action(t.act(obs),c.env.action_noise_std)
  obs,r,term,trunc,_=t.env.step(a)
  row=(t.buffer.ptr-1)%t.buffer.T
  t.wm.update_normalization(t.buffer.s[row],t.buffer.s1[row],t.buffer.a[row],t.buffer.xi[row])
  assert t.buffer.rows==i+1
  assert torch.isfinite(t.buffer.s1[row]).all()
  if (i+1)%16==0:print('collected',i+1,flush=True)
 torch.cuda.synchronize()
 t.env_step_count=64
 mark('collect64',seconds=time.perf_counter()-start,rows=t.buffer.rows,
      sgs_updates=int(t.command.sgs_sampler.total_updates))
 start=time.perf_counter();wm=t.wm_trainer.update(t.buffer);torch.cuda.synchronize()
 assert all(torch.isfinite(torch.tensor(v)) for v in wm.values()),wm
 mark('wm_default_update',seconds=time.perf_counter()-start,metrics=wm)
 start=time.perf_counter();pi=t.policy_trainer.update(t.buffer);torch.cuda.synchronize()
 assert all(torch.isfinite(torch.tensor(v)) for v in pi.values()),pi
 for model in (t.wm,t.policy,t.critic):
  assert all(torch.isfinite(p).all() for p in model.parameters())
 mark('policy_default_update',seconds=time.perf_counter()-start,metrics=pi)
 if t.world>1:
  import torch.distributed as dist
  x=torch.cat((t.wm.in_norm.mean,t.policy.obs_norm.mean,t.critic.norm.mean))
  ranks=[torch.empty_like(x) for _ in range(t.world)]
  dist.all_gather(ranks,x)
  for other in ranks:torch.testing.assert_close(x,other)
 mark('completed',world_size=t.world,peak_gib=torch.cuda.max_memory_allocated()/1024**3,
      reserved_gib=torch.cuda.memory_reserved()/1024**3,
      free_gib=torch.cuda.mem_get_info()[0]/1024**3)
 print('DEFAULT_SCALE_VALIDATION_OK',flush=True)
finally:
 t.env.close()
 if t.writer:t.writer.close()
 from supermimic.training.distributed import finalize
 finalize()
