"""Explicit GPU smoke; reduced network/batch sizes, unchanged 500-update phase boundary."""
import os
from pathlib import Path
import supermimic.scripts.train
import torch
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
c=TrainConfig();c.env.robot='p2';c.policy.fsq=True;c.env.num_envs=2;c.env.replay_capacity=128
c.data.npz_dir=Path(os.environ['PARITY_SMOKE_MOTIONS'])
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
t.run()
assert t.iteration==c.schedule.total_iterations
assert t.policy.obs_norm.count>0 and t.wm.in_norm.count>0
if t.iteration>500:assert t.critic.norm.count>0
assert torch.isfinite(t.policy.obs_norm.mean).all()
print('PARITY_TRAIN_OK',t.iteration,t.env_step_count,t.world,flush=True)
t.env.close()
