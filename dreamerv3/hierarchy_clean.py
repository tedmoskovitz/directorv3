import functools
import copy

import embodied
import numpy as np
import jax
import pdb
import jax.numpy as jnp
# from tensorflow_probability import distributions as tfd
from tensorflow_probability.substrates import jax as tfp

from . import agent
from . import expl
from . import nets
from . import ninjax as nj
from . import jaxutils


class FeudalHRL(nj.Module):

  def __init__(self, wm, act_space, config):
    self.wm = wm
    self.config = config
    self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[1:]
    self.skill_space = embodied.Space(
        np.int32 if config.goal_encoder.dist == 'onehot' else np.float32,
        config.skill_shape)

    ########################################################################
    #
    #
    #  WORKER CONFIG
    #
    #
    ########################################################################

    wconfig = config.update({
        'actor.inputs': self.config.worker_inputs,
        'critic.inputs': self.config.worker_inputs,
    })

    self.worker = agent.ImagActorCritic(
        {
            'extr':
                agent.VFunction(lambda s: s['reward_extr'],
                                wconfig,
                                name='worker_critic_extr'),
            'expl':
                agent.VFunction(lambda s: s['reward_expl'],
                                wconfig,
                                name='worker_critic_expl'),
            'goal':
                agent.VFunction(lambda s: s['reward_goal'],
                                wconfig,
                                name='worker_critic_goal'),
        },
        config.worker_rews,
        act_space,
        wconfig,
        name='worker')

    ########################################################################
    #
    #
    #  MANAGER CONFIG
    #
    #
    ########################################################################

    mconfig = config.update({
        'actor_grad_cont': 'reinforce',
    })

    self.manager = agent.ImagActorCritic(
        {
            'extr':
                agent.VFunction(lambda s: s['reward_extr'],
                                mconfig,
                                name='manager_critic_extr'),
            'expl':
                agent.VFunction(lambda s: s['reward_expl'],
                                mconfig,
                                name='manager_critic_expl'),
            'goal':
                agent.VFunction(lambda s: s['reward_goal'],
                                mconfig,
                                name='manager_critic_goal'),
        },
        config.manager_rews,
        self.skill_space,
        mconfig,
        name='manager')

    ########################################################################
    #
    #
    #  GOAL ENCODER CONFIG
    #
    #
    ########################################################################

    shape = self.skill_space.shape
    if self.skill_space.discrete:
      self.goal_prior = jaxutils.OneHotDist(jnp.zeros(shape))
      self.goal_prior = jaxutils.tfd.Independent(self.goal_prior,
                                                 len(shape) - 1)
    else:
      self.goal_prior = jaxutils.tfd.Normal(jnp.zeros(shape), jnp.ones(shape))
      self.goal_prior = jaxutils.tfd.Independent(self.goal_prior, len(shape))

    self.goal_shape = (self.config.rssm.deter,)
    self.goal_encoder = nets.MLP(config.skill_shape,
                                 dims='context',
                                 **config.goal_encoder,
                                 name='encoder_mlp')
    self.goal_decoder = nets.MLP(self.goal_shape,
                                 dims='context',
                                 **self.config.goal_decoder,
                                 name='decoder_mlp')
    self.target_goal_decoder = nets.MLP(self.goal_shape,
                                        dims='context',
                                        **self.config.goal_decoder,
                                        name='target_decoder_mlp')

    # this is to automatically update the KL scaling
    self.goal_vae_kl = jaxutils.AutoAdapt((),
                                          **self.config.encdec_kl,
                                          name='goal-vae_kl-adapt')

    # optimizer for goal autoencoder
    self.goal_encoder_opt = jaxutils.Optimizer(**config.encdec_opt,
                                               name='autoenc_opt')

    self.goal_decoder_context = nets.Input(['deter'])

    ########################################################################
    #
    #
    #  EXPLORATION CONFIG
    #
    #
    ########################################################################

    if self.config.expl_rew == 'disag':
      self.expl_reward = expl.Disag(wm, act_space, config)

    elif self.config.expl_rew == 'adver':
      self.expl_reward = self.elbo_reward
    else:
      raise NotImplementedError(self.config.expl_rew)

  def initial(self, batch_size):
    return {
        'step':
            jnp.zeros((batch_size,), jnp.int64),
        'skill':
            jnp.zeros((batch_size,) + self.config.skill_shape, jnp.float32),
        'goal':
            jnp.zeros((batch_size,) + self.goal_shape, jnp.float32),
        'mean_skill':
            jnp.zeros((batch_size,) + self.config.skill_shape, jnp.float32),
        'mean_goal':
            jnp.zeros((batch_size,) + self.goal_shape, jnp.float32),
    }

  def policy(self, latent, carry, imag=False):
    """HRL policy executing the actions from the worker."""
    duration = self.config.train_skill_duration if imag else (
        self.config.env_skill_duration)
    update = (carry['step'] % duration) == 0

    # Switch variable controls the following:
    #   * If we switch -> use newly sampled goal / skill.
    #   * If we don't switch -> use the old one.
    def switch(x, y):
      return (jnp.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
              jnp.einsum('i,i...->i...', update.astype(x.dtype), y))

    # TODO(ag): This needs to be less hard-coded
    manager_inputs = copy.deepcopy(latent)

    # Skill is 'z' sampled from Manager.
    # skill is [batch_length * batch_size, *skill_shape]
    sampled_skill = self.manager.actor(
        jaxutils.sg(manager_inputs)).sample(seed=nj.rng())
    mean_skill = self.manager.actor(jaxutils.sg(manager_inputs)).mean()

    skill = jaxutils.sg(switch(carry['skill'], sampled_skill))
    mean_skill = jaxutils.sg(switch(carry['mean_skill'], mean_skill))

    new_goal = self.target_goal_decoder({
        'skill': skill,
        'context': self.goal_decoder_context(latent)
    }).mode()
    new_mean_goal = self.target_goal_decoder({
        'skill': mean_skill,
        'context': self.goal_decoder_context(latent)
    }).mode()

    # setting manager_delta == True means that the goal is really a delta
    # in state rather than a proposed state
    if self.config.manager_delta:
      new_goal = self.goal_decoder_context(latent).astype(
          jnp.float32) + new_goal
      new_mean_goal = self.goal_decoder_context(latent).astype(
          jnp.float32) + new_mean_goal
    # Goal is 'g' decoded from worker
    goal = jaxutils.sg(switch(carry['goal'], new_goal))
    mean_goal = jaxutils.sg(switch(carry['mean_goal'], new_mean_goal))
    delta = goal - self.goal_decoder_context(latent).astype(jnp.float32)
    mean_delta = mean_goal - self.goal_decoder_context(latent).astype(
        jnp.float32)
    dist = self.worker.actor(
        jaxutils.sg({
            **latent,
            'goal': goal,
            'delta': delta,
            'mean_goal': mean_goal,
            'mean_delta': mean_delta,
        }))
    # dist is a jaxutils dist with shape [batch_length * batch_size,
    # action_dim]
    outs = {'action': dist}
    if 'image' in self.wm.heads['decoder'].shapes:
      outs['log_goal'] = self.wm.heads['decoder']({
          'deter': goal,
          'stoch': self.wm.rssm.get_stoch(goal),
      })['image'].mode()
      outs['log_mean_goal'] = self.wm.heads['decoder']({
          'deter': mean_goal,
          'stoch': self.wm.rssm.get_stoch(mean_goal),
      })['image'].mode()
    carry = {
        'step': carry['step'] + 1,
        'skill': skill,
        'goal': goal,
        'mean_goal': mean_goal,
        'mean_skill': mean_skill
    }
    # outs = {'action': action_dist (batch_size * batch_length, *action.shape)}
    # carry['skill'] is [batch_size*batch_length, *skill.shape]
    # carry['goal'] is [batch_size*batch_length, rssm.deter size]
    return outs, carry

  def train(self, imagine, start, data):
    """Trains the HRL agent"""

    def success(rew):
      return (rew[-1] > 0.7).astype(jnp.float32).mean()

    metrics = {}
    if self.config.expl_rew == 'disag':
      metrics.update(self.expl_reward.train(data))

    # Train goal auto-encoder from replay data.
    if self.config.vae_replay:
      metrics.update(self.train_goal_vae_from_replay(data))

    if self.config.jointly == 'new':
      traj, mets = self.train_jointly(start)
      metrics.update(mets)
      metrics['success_manager'] = success(traj['reward_goal'])
      if self.config.vae_imag:
        metrics.update(self.train_goal_vae_from_wm(start))
    elif self.config.jointly == 'off':
      for impl in self.config.worker_goals:
        goal = self.propose_goal(start, impl)
        traj, mets = self.train_worker(start, goal)
        metrics.update(mets)
        metrics[f'success_{impl}'] = success(traj['reward_goal'])
      traj, mets = self.train_manager(start)
      metrics.update(mets)
      metrics['success_manager'] = success(traj['reward_goal'])
    else:
      raise NotImplementedError(self.config.jointly)
    return None, metrics

  def train_jointly(self, start):
    start = start.copy()
    policy = functools.partial(self.policy, imag=True)
    # carry is an HRL dict containing 'step', 'skill', and 'goal'
    traj = self.wm.imagine_carry(policy, start, self.config.imag_horizon,
                                 self.initial(len(start['is_first'])))

    # Add rewards to the trajectory.
    traj['reward_extr'] = self.extr_reward(traj)
    traj['reward_expl'] = self.expl_reward(traj)
    traj['reward_goal'] = self.goal_reward(traj)
    traj['delta'] = traj['goal'] - self.goal_decoder_context(traj).astype(
        jnp.float32)

    def worker_loss(traj):
      wtraj = self.split_traj(traj)
      worker_loss, worker_metrics = self.worker.loss(wtraj)
      worker_metrics = {"worker_" + k: v for k, v in worker_metrics.items()}
      return worker_loss, (traj, worker_metrics, wtraj)

    def manager_loss(traj):
      mtraj = self.abstract_traj(traj)
      manager_loss, manager_metrics = self.manager.loss(mtraj)
      manager_metrics = {"manager_" + k: v for k, v in manager_metrics.items()}
      return manager_loss, (traj, manager_metrics, mtraj)

    ########################################################################
    #
    # WORKER TRAINING
    #
    ########################################################################

    # update the worker actor
    wmets, (traj, worker_metrics, wtraj) = self.worker.opt(self.worker.actor,
                                                           worker_loss,
                                                           traj,
                                                           has_aux=True)
    worker_metrics.update(wmets)
    # update the worker critic
    for key, critic in self.worker.critics.items():
      cwmets = critic.train(wtraj, self.worker.actor)
      worker_metrics.update({
          f'{key}_worker-critic_{k}': v for k, v in cwmets.items()
      })
    worker_metrics['worker_actor_opt_loss'] = worker_metrics['actor_opt_loss']

    ########################################################################
    #
    # MANAGER TRAINING
    #
    ########################################################################
    # update the manager actor
    mmets, (traj, manager_metrics, mtraj) = self.manager.opt(self.manager.actor,
                                                             manager_loss,
                                                             traj,
                                                             has_aux=True)
    manager_metrics.update(mmets)
    # update the manager critic
    for key, critic in self.manager.critics.items():
      mmets = critic.train(mtraj, self.manager.actor)
      manager_metrics.update({
          f'{key}_manager-critic_{k}': v for k, v in mmets.items()
      })
    manager_metrics['manager_actor_opt_loss'] = manager_metrics[
        'actor_opt_loss']

    return traj, {**worker_metrics, **manager_metrics}

  def train_manager(self, start):
    start = start.copy()
    policy = functools.partial(self.policy, imag=True)
    # carry is an HRL dict containing 'step', 'skill', and 'goal'
    traj = self.wm.imagine_carry(policy, start, self.config.imag_horizon,
                                 self.initial(len(start['is_first'])))
    traj['reward_extr'] = self.extr_reward(traj)
    traj['reward_expl'] = self.expl_reward(traj)
    traj['reward_goal'] = self.goal_reward(traj)
    traj['delta'] = traj['goal'] - self.goal_decoder_context(traj).astype(
        jnp.float32)

    def manager_loss(traj):
      mtraj = self.abstract_traj(traj)
      manager_loss, manager_metrics = self.manager.loss(mtraj)
      manager_metrics = {"manager_" + k: v for k, v in manager_metrics.items()}
      return manager_loss, (traj, manager_metrics, mtraj)

    # update the manager actor
    mmets, (traj, manager_metrics, mtraj) = self.manager.opt(self.manager.actor,
                                                             manager_loss,
                                                             traj,
                                                             has_aux=True)
    manager_metrics.update(mmets)
    # update the manager critic
    for key, critic in self.manager.critics.items():
      mmets = critic.train(mtraj, self.manager.actor)
      manager_metrics.update({
          f'{key}_manager-critic_{k}': v for k, v in mmets.items()
      })
    manager_metrics['manager_actor_opt_loss'] = manager_metrics[
        'actor_opt_loss']
    return traj, manager_metrics

  def train_worker(self, start, goal):

    def worker(s):
      return self.worker.actor(
          jaxutils.sg({
              **s,
              'goal': goal,
              'delta': goal - self.goal_decoder_context(s).astype(jnp.float32),
          })).sample(seed=nj.rng())

    traj = self.wm.imagine_carry(worker, start, self.config.imag_horizon,
                                 self.initial(len(start['is_first'])))
    traj['reward_extr'] = self.extr_reward(traj)
    traj['reward_expl'] = self.expl_reward(traj)
    traj['reward_goal'] = self.goal_reward(traj)
    traj['delta'] = traj['goal'] - self.goal_decoder_context(traj).astype(
        jnp.float32)

    def worker_loss(traj):
      wtraj = self.split_traj(traj)
      worker_loss, worker_metrics = self.worker.loss(wtraj)
      worker_metrics = {"worker_" + k: v for k, v in worker_metrics.items()}
      return worker_loss, (traj, worker_metrics, wtraj)

    # update the worker actor
    wmets, (traj, worker_metrics, wtraj) = self.worker.opt(self.worker.actor,
                                                           worker_loss,
                                                           traj,
                                                           has_aux=True)
    worker_metrics.update(wmets)
    # update the worker critic
    for key, critic in self.worker.critics.items():
      cwmets = critic.train(wtraj, self.worker.actor)
      worker_metrics.update({
          f'{key}_worker-critic_{k}': v for k, v in cwmets.items()
      })
    worker_metrics['worker_actor_opt_loss'] = worker_metrics['actor_opt_loss']

    return traj, worker_metrics

  def train_goal_vae_from_wm(self, start):
    start = start.copy()
    policy = functools.partial(self.policy, imag=True)
    traj = self.wm.imagine_carry(policy, start, self.config.imag_horizon,
                                 self.initial(len(start['is_first'])))
    return self._train_goal_vae(traj, prefix='wm')

  def train_goal_vae_from_replay(self, data):
    return self._train_goal_vae(data, prefix='replay')

  def _train_goal_vae(self, data, prefix=''):
    metrics = {}
    feat = self.goal_decoder_context(data).astype(jnp.float32)
    if 'context' in self.config.goal_decoder.inputs:
      if self.config.vae_span:
        context = feat[:, 0]
        goal = feat[:, -1]
      else:
        assert feat.shape[1] > self.config.train_skill_duration
        context = feat[:, :-self.config.train_skill_duration]
        goal = feat[:, self.config.train_skill_duration:]
    else:
      goal = context = feat
    # enc is tensorflow_probability.substrates.jax.distributions.independent.Independent
    # goal is (batch_size, batch_length, goal_dim ( = rssm.deter))

    def loss_fn(goal, context):
      local_metrics = {}
      enc = self.goal_encoder({'goal': goal, 'context': context})
      dec = self.goal_decoder({
          'skill': enc.sample(seed=nj.rng()),
          'context': context
      })
      rec = -dec.log_prob(jaxutils.sg(goal))
      if self.config.goal_kl:
        kl = jaxutils.tfd.kl_divergence(enc, self.goal_prior)
        if self.config.use_fixed_kl:
          kl *= self.config.fixed_kl_coeff
        else:
          kl, mets = self.goal_vae_kl(kl)
          local_metrics.update({
              f'{prefix}/goalkl_{k}': v for k, v in mets.items()
          })
        assert rec.shape == kl.shape, (rec.shape, kl.shape)
      else:
        kl = 0.0

      local_metrics[f'{prefix}/goalrec_mean'] = rec.mean()
      local_metrics[f'{prefix}/goalrec_std'] = rec.std()
      local_metrics[f'{prefix}/goal_kl'] = jnp.mean(kl)
      local_metrics[f'{prefix}/goal_total_loss'] = (rec + kl).mean()

      return (rec + kl).mean(), local_metrics

    mets, local_metrics = self.goal_encoder_opt(
        [self.goal_encoder, self.goal_decoder],
        loss_fn,
        goal,
        context,
        has_aux=True)
    metrics.update(mets)
    metrics.update(local_metrics)

    return metrics

  def propose_goal(self, start, impl):
    feat = self.goal_decoder_context(start).astype(jnp.float32)
    if impl == 'replay':
      target = jax.random.shuffle(nj.rng(), feat).astype(jnp.float32)
      skill = self.goal_encoder({
          'goal': target,
          'context': feat
      }).sample(seed=nj.rng())
      mean_skill = self.goal_encoder({'goal': target, 'context': feat}).mean()
      goal = self.target_goal_decoder({'skill': skill, 'context': feat}).mode()
      mean_goal = self.target_goal_decoder({
          'skill': mean_skill,
          'context': feat
      }).mode()
      return goal, mean_goal
    elif impl == 'replay_direct':
      goal = jax.random.shuffle(nj.rng(), feat).astype(jnp.float32)
      return goal, goal
    elif impl == 'manager':
      # TODO(ag): This needs to be less hard-coded
      manager_inputs = copy.deepcopy(start)
      skill = self.manager.actor(manager_inputs).sample(seed=nj.rng())
      mean_skill = self.manager.actor(manager_inputs).mean()
      goal = self.target_goal_decoder({'skill': skill, 'context': feat}).mode()
      mean_goal = self.target_goal_decoder({
          'skill': mean_skill,
          'context': feat
      }).mode()
      goal = feat + goal if self.config.manager_delta else goal
      mean_goal = feat + mean_goal if self.config.manager_delta else goal
      return goal, mean_goal
    elif impl == 'prior':
      skill = self.goal_prior.sample(len(start['is_terminal']), seed=nj.rng())
      # TODO(ag): double check shapes
      mean_skill = self.goal_prior.mean()
      goal = self.target_goal_decoder({'skill': skill, 'context': feat}).mode()
      mean_goal = self.target_goal_decoder({
          'skill': skill,
          'context': feat
      }).mode()
      return goal, mean_goal
    else:
      raise NotImplementedError(impl)

  def goal_reward(self, traj):
    feat = self.goal_decoder_context(traj).astype(jnp.float32)
    goal = jaxutils.sg(traj['goal'].astype(jnp.float32))
    skill = jaxutils.sg(traj['skill'].astype(jnp.float32))
    mean_goal = jaxutils.sg(traj['mean_goal'].astype(jnp.float32))
    mean_skill = jaxutils.sg(traj['mean_skill'].astype(jnp.float32))
    context = jaxutils.sg(
        jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0))
    if self.config.goal_reward == 'dot':
      return jnp.einsum('...i,...i->...', goal, feat)[1:]
    elif self.config.goal_reward == 'dir':
      return jnp.einsum('...i,...i->...', jnp.nn.l2_normalize(goal, -1),
                        feat)[1:]
    elif self.config.goal_reward == 'normed_inner':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'normed_squared':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      return -((goal / norm - feat / norm)**2).mean(-1)[1:]
    elif self.config.goal_reward == 'cosine_lower':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
    elif self.config.goal_reward == 'cosine_lower_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.maximum(gnorm, fnorm)
      cos = jnp.einsum('...i,...i->...', goal / gnorm, feat / fnorm)[1:]
      return jnp.nn.relu(cos)
    elif self.config.goal_reward == 'cosine_frac':
      gnorm = jnp.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = jnp.einsum('...i,...i->...', goal, feat)
      mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
      return (cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_frac_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1) + 1e-12
      goal /= gnorm[..., None]
      feat /= fnorm[..., None]
      cos = jnp.einsum('...i,...i->...', goal, feat)
      mag = jnp.minimum(gnorm, fnorm) / jnp.maximum(gnorm, fnorm)
      return jnp.nn.relu(cos * mag)[1:]
    elif self.config.goal_reward == 'cosine_max':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'cosine_max_diff':
      feat = feat[1:] - feat[:-1]
      goal = goal[1:]
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)
    elif self.config.goal_reward == 'cosine_max_mean':
      gnorm = jnp.linalg.norm(mean_goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
    elif self.config.goal_reward == 'cosine_max_pos':
      gnorm = jnp.linalg.norm(goal, axis=-1, keepdims=True) + 1e-12
      fnorm = jnp.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12
      norm = jnp.maximum(gnorm, fnorm)
      cos = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return jnp.nn.relu(cos)
    elif self.config.goal_reward == 'normed_inner_clip':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return jnp.clip_by_value(cosine, -1.0, 1.0)
    elif self.config.goal_reward == 'normed_inner_clip_pos':
      norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
      cosine = jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
      return jnp.clip_by_value(cosine, 0.0, 1.0)
    elif self.config.goal_reward == 'diff':
      goal = goal[:-1] / jnp.linalg.norm(
          goal[:-1], ord=2, axis=-1, keepdims=True)
      diff = jnp.concatenate([feat[1:] - feat[:-1]], 0)
      return jnp.einsum('...i,...i->...', goal, diff)
    elif self.config.goal_reward == 'norm':
      return -jnp.linalg.norm(goal - feat, axis=-1)[1:]
    elif self.config.goal_reward == 'squared':
      return -((goal - feat)**2).sum(-1)[1:]
    elif self.config.goal_reward == 'epsilon':
      return ((goal - feat).mean(-1) < 1e-3).astype(jnp.float32)[1:]
    elif self.config.goal_reward == 'enclogprob':
      return self.goal_encoder({
          'goal': goal,
          'context': context
      }).log_prob(skill)[1:]
    elif self.config.goal_reward == 'encprob':
      return self.goal_encoder({
          'goal': goal,
          'context': context
      }).prob(skill)[1:]
    elif self.config.goal_reward == 'enc_normed_cos':
      dist = self.goal_encoder({'goal': goal, 'context': context})
      probs = dist.distribution.probs_parameter()
      norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
      return jnp.einsum('...ij,...ij->...', probs / norm, skill / norm)[1:]
    elif self.config.goal_reward == 'enc_normed_squared':
      dist = self.goal_encoder({'goal': goal, 'context': context})
      probs = dist.distribution.probs_parameter()
      norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
      return -((probs / norm - skill / norm)**2).mean([-2, -1])[1:]
    else:
      raise NotImplementedError(self.config.goal_reward)

  def elbo_reward(self, traj):
    feat = self.goal_decoder_context(traj).astype(jnp.float32)
    context = jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
    enc = self.goal_encoder({'goal': feat, 'context': context})
    dec = self.target_goal_decoder({
        'skill': enc.sample(seed=nj.rng()),
        'context': context
    })
    if self.config.adver_impl == 'abs':
      return jnp.abs(dec.mode() - feat).mean(-1)[1:]
    elif self.config.adver_impl == 'squared':
      return ((dec.mode() - feat)**2).mean(-1)[1:]
    ll = dec.log_prob(feat)
    kl = jaxutils.tfd.kl_divergence(enc, self.goal_prior)
    if self.config.adver_impl == 'elbo_scaled':
      return (kl - ll / self.goal_vae_kl.scale())[1:]
    elif self.config.adver_impl == 'elbo_unscaled':
      return (kl - ll)[1:]
    raise NotImplementedError(self.config.adver_impl)

  def split_traj(self, traj):
    traj = traj.copy()
    k = self.config.train_skill_duration

    # Q: Why do we need this assert ?
    # assert len(traj['action']) % k == 1

    def reshape(x):
      return x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))

    for key, val in list(traj.items()):
      val = jnp.concatenate([0 * val[:1], val], 0) if 'reward' in key else val
      # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
      val = jnp.concatenate([reshape(val[:-1]), val[k::k][:, None]], 1)
      # N val K val B val F... -> K val (N B) val F...
      val = val.transpose([1, 0] + list(range(2, len(val.shape))))
      val = val.reshape([val.shape[0], np.prod(val.shape[1:3])] +
                        list(val.shape[3:]))
      val = val[1:] if 'reward' in key else val
      traj[key] = val
    # Bootstrap sub trajectory against current not next goal.
    traj['goal'] = jnp.concatenate([traj['goal'][:-1], traj['goal'][:1]], 0)
    traj['mean_goal'] = jnp.concatenate(
        [traj['mean_goal'][:-1], traj['mean_goal'][:1]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def abstract_traj(self, traj):
    traj = traj.copy()
    # So, the 'skill' in high-level means the high-level action
    traj['action'] = traj.pop('skill')
    k = self.config.train_skill_duration

    # We chunk data by skill duration because every manager "timestep"
    # is really k timesteps
    def reshape(x):
      return x.reshape([x.shape[0] // k, k] + list(x.shape[1:]))

    weights = jnp.cumprod(reshape(traj['cont'][:-1]), 1)
    for key, value in list(traj.items()):
      if 'reward' in key:
        traj[key] = (reshape(value) * weights).mean(1)
      elif key == 'cont':
        traj[key] = jnp.concatenate([value[:1], reshape(value[1:]).prod(1)], 0)
      else:
        traj[key] = jnp.concatenate([reshape(value[:-1])[:, 0], value[-1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    metrics = {}
    for impl in ('manager', 'prior', 'replay'):
      # for manager, make a video of the decoded goals, for prior it's
      # samples of z from the prior then decoded, for replay it's the
      # goal ae
      for key, video in self.report_worker(data, impl).items():
        metrics[f'impl_{impl}_{key}'] = video
    return metrics

  def report_worker(self, data, impl):
    # Prepare initial state.
    # data['observation] is (batch_size, batch_length, observation_dim)
    decoder = self.wm.heads['decoder']
    # take first 6 samples from the batch (Why?)
    states, _ = self.wm.rssm.observe(
        self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
    start = {k: v[:, 4] for k, v in states.items()}
    start['is_terminal'] = data['is_terminal'][:6, 4]
    goal, mean_goal = self.propose_goal(start, impl)

    # goal is (6, batch_length, state_dim)
    # self.feat(states) is the same shape as goal
    # Worker rollout.

    def worker(s):
      return self.worker.actor({
          **s,
          'goal':
              goal,
          'delta':
              goal - self.goal_decoder_context(s).astype(jnp.float32),
          'mean_goal':
              mean_goal,
          'mean_delrta':
              mean_goal - self.goal_decoder_context(s).astype(jnp.float32),
      }).sample(seed=nj.rng())

    traj = self.wm.imagine(worker, start, self.config.worker_report_horizon)
    # Decoder into images.
    initial = decoder(start)
    target = decoder({'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})
    rollout = decoder(traj)
    # Stich together into videos. Currently, rollout['observation'] is a
    # SymLogDist; have mean(), mode(); they're [B, T, D] with the tabular
    # envs
    videos = {}
    for k in rollout.keys():
      if k not in decoder.cnn_shapes and k not in ['observation']:
        continue
      if k in decoder.cnn_shapes:
        length = 1 + self.config.worker_report_horizon
        rows = []
        rows.append(jnp.repeat(initial[k].mode()[:, None], length, 1))
        if target is not None:
          rows.append(jnp.repeat(target[k].mode()[:, None], length, 1))
        rows.append(rollout[k].mode().transpose((1, 0, 2, 3, 4)))
        # input to video_grid should be B, T, H, W, C

        videos[k] = jaxutils.video_grid(jnp.concatenate(rows, 2))
      elif k == 'observation':
        # assume size [B, T, D]
        # for T-Maze. D = 10. First 8 dims are the env states, final two
        # is a one-hot representation of the goal.
        # For now, let's just reshape to [B, T, 1, D, 1]
        B, T, D = rollout[k].mode().shape
        frame_extracts = rollout[k].mode().reshape(B, T, 1, D, 1)
        t_central = frame_extracts[:, :, 0, :4:]
        t_left = frame_extracts[:, :, 0, 4:6, :]
        t_right = frame_extracts[:, :, 0, 6:8, :]
        frame = -jnp.ones((B, T, 6, 7, 1))
        # update blank_frame with the goal representation
        frame = frame.at[:, :, 1:5, 3, :].set(t_central)
        frame = frame.at[:, :, 1, 1:3, :].set(t_left)
        frame = frame.at[:, :, 1, 4:6, :].set(t_right)
        if frame.shape[-1] == 1:
          frame = jnp.repeat(frame, 3, -1)
        videos[k] = jaxutils.video_grid(frame)

    return videos
