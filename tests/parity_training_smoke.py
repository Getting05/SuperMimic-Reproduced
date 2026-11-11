"""Explicit GPU smoke; reduced network/batch sizes, unchanged 500-update phase boundary."""
import os
from pathlib import Path
import supermimic.scripts.train
import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
c=TrainConfig();c.env.robot='p2';c.policy.fsq=True;c.env.num_envs=2;c.env.replay_capacity=128
c.data.npz_dir=Path(os.environ.get('PARITY_SMOKE_MOTIONS','data/npz/lafan1'))
if os.environ.get('PARITY_SMOKE_LIBRARY'):c.data.library=Path(os.environ['PARITY_SMOKE_LIBRARY'])
c.data.library_device=os.environ.get('PARITY_SMOKE_LIBRARY_DEVICE','gpu')
c.log_root=Path(os.environ.get('PARITY_SMOKE_OUTPUT','/tmp/parity-final-smoke'))
c.schedule.total_iterations=int(os.environ.get('PARITY_SMOKE_UPDATES','501'))
c.schedule.log_interval=100
c.wm_trainer.batches_per_update=1;c.wm_trainer.batch_size=4;c.wm_trainer.compile=None
c.world_model.hidden=(32,32,32);c.policy.encoder_hidden=(32,);c.policy.decoder_hidden=(32,)
c.policy_trainer.batch_size=4;c.policy_trainer.horizon=32;c.policy_trainer.critic_epochs=1
c.policy_trainer.critic_minibatches=1;c.policy_trainer.compile=None
c.env.sampling_mode=os.environ.get('PARITY_SMOKE_SAMPLER','return')
if os.environ.get('PARITY_SMOKE_RESUME'):
 c.resume=c.log_root/'debug/ckpt_latest.pt'
 c.schedule.total_iterations+=2
t=Trainer(c)
if os.environ.get("PARITY_SMOKE_JOINT"):
 t.iteration=t.num_updates=500
 c.schedule.total_iterations=501
if t.world>1:
 import torch.distributed as dist
 original=t.save
 def checked_save(name="ckpt_latest.pt"):
  x=torch.cat((t.policy.obs_norm.mean,t.wm.in_norm.mean,t.policy.obs_norm.count[None]))
  values=[torch.empty_like(x) for _ in range(t.world)]
  dist.all_gather(values,x)
  for other in values:torch.testing.assert_close(x,other)
  original(name)
 t.save=checked_save
t.run()
assert t.iteration==c.schedule.total_iterations
assert t.policy.obs_norm.count>0 and t.wm.in_norm.count>0
if t.iteration>500:assert t.critic.norm.count>0
assert torch.isfinite(t.policy.obs_norm.mean).all()
for m in (t.wm,t.policy,t.critic):
 assert all(torch.isfinite(p).all() for p in m.parameters())
print('PARITY_TRAIN_OK',t.iteration,t.env_step_count,t.world,flush=True)
t.env.close()
