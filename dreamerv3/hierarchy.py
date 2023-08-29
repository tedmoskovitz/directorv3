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


class Hierarchy(nj.Module):

    def __init__(self, wm, act_space, config):
        self.wm = wm
        self.config = config
        self.short_circuit_worker = config.get("short_circuit_worker", False)
        if self.short_circuit_worker:
            assert np.prod(config.skill_shape) == np.prod(act_space.shape), "Incompatible skill shape for short-circuiting" 
            assert not config.goal_kl, "Goal KL must be off for short-circuiting."
            assert config.jointly == 'off', "Must separate training for short-circuiting."
        self.extr_reward = lambda traj: self.wm.heads['reward'](traj).mean()[
            1:]
        self.skill_space = embodied.Space(
            np.int32 if config.goal_encoder.dist == 'onehot' else np.float32,
            config.skill_shape)

        wconfig = config.update({
            'actor.inputs': self.config.worker_inputs,
            'critic.inputs': self.config.worker_inputs,
        })

        self.worker = agent.ImagActorCritic({
            'extr': agent.VFunction(lambda s: s['reward_extr'], wconfig, name='worker_critic_extr'),
            'expl': agent.VFunction(lambda s: s['reward_expl'], wconfig, name='worker_critic_expl'),
            'goal': agent.VFunction(lambda s: s['reward_goal'], wconfig, name='worker_critic_goal'),
        }, config.worker_rews, act_space, wconfig, name='worker')

        ########################################################################
        #
        # STATE ABSTRACTION
        #
        ########################################################################

        mconfig = config.update({
            'actor_grad_cont': 'reinforce',
            # 'actent.target': config.manager_actent,  # don't need this in v3, used for tuned entropy coeff
        })

        # State Encoder for State Abstraction
        if self.config.use_state_abstraction:
            self.abstract_state_feat = nets.Input(self.config.abstract_state_observations)
            self.state_abstraction_encoder = nets.MLP(
                self.config.state_abstraction_shape, dims='context', **self.config.state_abstraction_encoder, name='state_abstraction_encoder_mlp')
            self.state_abstraction_decoder = nets.MLP(
                (self.config.rssm.deter, ), dims='context', **self.config.state_abstraction_decoder, name='state_abstraction_decoder_mlp')
            self.state_abstraction_kl = jaxutils.AutoAdapt(
                (), **self.config.state_abstraction_kl, name='state_abstraction-vae_kl-adapt')
            self.state_abstraction_kl_opt = jaxutils.Optimizer(
                **config.state_abstraction_opt, name='state_abstraction_opt')
            # Prior
            shape = self.config.state_abstraction_shape
            self.state_abstraction_prior = jaxutils.OneHotDist(
                jnp.zeros(shape))
            self.state_abstraction_prior = jaxutils.tfd.Independent(
                self.state_abstraction_prior, len(shape) - 1)
            
            # Override Manager inputs
            mconfig.update({
                'actor.inputs': ['abstract_state',],
                'critic.inputs': ['abstract_state', ],
            })                 
        
        self.manager = agent.ImagActorCritic({
            'extr': agent.VFunction(lambda s: s['reward_extr'], mconfig, name='manager_critic_extr'),
            'expl': agent.VFunction(lambda s: s['reward_expl'], mconfig, name='manager_critic_expl'),
            'goal': agent.VFunction(lambda s: s['reward_goal'], mconfig, name='manager_critic_goal'),
        }, config.manager_rews, self.skill_space, mconfig, name='manager')
        if self.config.expl_rew == 'disag':
            self.expl_reward = expl.Disag(wm, act_space, config)
            
        elif self.config.expl_rew == 'adver':
            self.expl_reward = self.elbo_reward
        else:
            raise NotImplementedError(self.config.expl_rew)
        if config.explorer:
            self.explorer = agent.ImagActorCritic({
                'expl': agent.VFunction(self.expl_reward, config, name='explorer_critic'),
            }, {'expl': 1.0}, act_space, config, name='explorer')

        shape = self.skill_space.shape
        if self.skill_space.discrete:
            self.prior = jaxutils.OneHotDist(jnp.zeros(shape))
            self.prior = jaxutils.tfd.Independent(self.prior, len(shape) - 1)
        else:
            self.prior = jaxutils.tfd.Normal(jnp.zeros(shape), jnp.ones(shape))
            self.prior = jaxutils.tfd.Independent(self.prior, len(shape))

        self.feat = nets.Input(['deter'])
        self.goal_shape = (self.config.rssm.deter,)
        self.enc = nets.MLP(
            config.skill_shape, dims='context', **config.goal_encoder, name='encoder_mlp')
        self.dec = nets.MLP(
            self.goal_shape, dims='context', **self.config.goal_decoder, name='decoder_mlp')
        # this is to automatically update the KL scaling
        self.kl = jaxutils.AutoAdapt(
            (), **self.config.encdec_kl, name='goal-vae_kl-adapt')
        # optimizer for goal autoencoder
        self.opt = jaxutils.Optimizer(**config.encdec_opt, name='autoenc_opt')
        # update counter
        self._update_count = nj.Variable(jnp.zeros, (), jnp.int64, name='update_count')
        self._alternate_frequency = -1 if not config.alternate_frequency else config.alternate_frequency
        train_manager_init = jnp.ones if self._alternate_frequency < 0 else jnp.zeros
        self._train_manager = nj.Variable(train_manager_init, (), bool, name='train_manager')
        self._train_worker = nj.Variable(jnp.ones, (), bool, name='train_worker')

    def initial(self, batch_size):
        return {
            'step': jnp.zeros((batch_size,), jnp.int64),
            'skill': jnp.zeros((batch_size,) + self.config.skill_shape, jnp.float32),
            'goal': jnp.zeros((batch_size,) + self.goal_shape, jnp.float32),
        }

    def policy(self, latent, carry, imag=False):
        # Q: Why it is called latent ?
        duration = self.config.train_skill_duration if imag else (
            self.config.env_skill_duration)
        update = (carry['step'] % duration) == 0
        # Switch variable controls the following:
        #   * If we switch -> use newly sampled goal / skill.
        #   * If we don't switch -> use the old one.
        def switch(x, y): return (
            jnp.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
            jnp.einsum('i,i...->i...', update.astype(x.dtype), y))
        
        # TODO(ag): This needs to be less hard-coded    
        manager_inputs = copy.deepcopy(latent)
        if self.config.use_state_abstraction:    
          state = context = self.abstract_state_feat(latent)
          state_abstraction_dist = self.state_abstraction_encoder({'state': state, 'context': context})
          abstract_state = state_abstraction_dist.sample(seed=nj.rng())
          manager_inputs['abstract_state'] = abstract_state


        if self.short_circuit_worker:
            # note that automatically "switch" every step in this case
            dist = self.manager.actor(jaxutils.sg(manager_inputs))
            carry = {'step': carry['step'] + 1, 'skill': carry['skill'], 'goal': carry['goal']}
            return {'action': dist}, carry

        # Skill is 'z' sampled from Manager.
        # skill is [batch_length * batch_size, *skill_shape]
        skill = jaxutils.sg(switch(carry['skill'],
                                   self.manager.actor(jaxutils.sg(manager_inputs)).sample(seed=nj.rng())))
        new_goal = self.dec(
            {'skill': skill, 'context': self.feat(latent)}).mode()
        # setting manager_delta == True means that the goal is really a delta
        # in state rather than a proposed state
        new_goal = (
            self.feat(latent).astype(jnp.float32) + new_goal
            if self.config.manager_delta else new_goal)
        # Goal is 'g' decoded from worker
        goal = jaxutils.sg(switch(carry['goal'], new_goal))
        # Delta is a difference between the goal and the latent.
        # Q: Why do we need this?
        delta = goal - self.feat(latent).astype(jnp.float32)
        dist = self.worker.actor(jaxutils.sg(
            {**latent, 'goal': goal, 'delta': delta}))
        # dist is a jaxutils dist with shape [batch_length * batch_size, action_dim]
        outs = {'action': dist}
        if 'image' in self.wm.heads['decoder'].shapes:
            outs['log_goal'] = self.wm.heads['decoder']({
                'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal),
            })['image'].mode()
        carry = {'step': carry['step'] + 1, 'skill': skill, 'goal': goal}
        # outs = {'action': action_dist (batch_size * batch_length, *action.shape)}
        # carry['skill'] is [batch_size*batch_length, *skill.shape]
        # carry['goal'] is [batch_size*batch_length, rssm.deter size]
        return outs, carry

    def train(self, imagine, start, data):
        # increment update count
        self._update_count.write(self._update_count.read() + 1)
        def flip_train(train_manager, train_worker):
            return jnp.logical_not(train_manager), jnp.logical_not(train_worker)
        def nothing(train_manager, train_worker):
            return train_manager, train_worker
        train_manager, train_worker = jax.lax.cond(
            self._alternate_frequency > 0 and jnp.mod(self._update_count.read(), self._alternate_frequency) == 0,
            flip_train, nothing, self._train_manager.read(), self._train_worker.read())
            
        self._train_manager.write(train_manager)
        self._train_worker.write(train_worker)
        def success(rew): return (rew[-1] > 0.7).astype(jnp.float32).mean()
        metrics = {}
        if self.config.expl_rew == 'disag':
            metrics.update(self.expl_reward.train(data))

        # Train goal auto-encoder.
        if self.config.vae_replay:
            metrics.update(self.train_vae_replay(data))

        # Train state abstraction.
        if self.config.use_state_abstraction:
            metrics.update(self.train_state_abstraction(data))

        if self.config.explorer:
            traj, mets = self.explorer.train(imagine, start, data)
            metrics.update({f'explorer_{k}': v for k, v in mets.items()})
            metrics.update(self.train_vae_imag(traj))
            if self.config.explorer_repeat:
                goal = self.feat(traj)[-1]
                metrics.update(self.train_worker(imagine, start, goal)[1])
        if self.config.jointly == 'new':
            traj, mets = self.train_jointly(
                imagine, start,
                train_manager=train_manager, train_worker=train_worker)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))
        elif self.config.jointly == 'old':
            traj, mets = self.train_jointly_old(imagine, start)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
            if self.config.vae_imag:
                metrics.update(self.train_vae_imag(traj))
        elif self.config.jointly == 'off':
            if not self.short_circuit_worker:
                for impl in self.config.worker_goals:
                    goal = self.propose_goal(start, impl)
                    traj, mets = self.train_worker(imagine, start, goal)
                    metrics.update(mets)
                    metrics[f'success_{impl}'] = success(traj['reward_goal'])
                    if self.config.vae_imag:
                        metrics.update(self.train_vae_imag(traj))
            traj, mets = self.train_manager(imagine, start)
            metrics.update(mets)
            metrics['success_manager'] = success(traj['reward_goal'])
        else:
            raise NotImplementedError(self.config.jointly)
        return None, metrics

    def train_jointly(self, imagine, start, train_manager=True, train_worker=True):
        start = start.copy()
        policy = functools.partial(self.policy, imag=True)
        # carry is an HRL dict containing 'step', 'skill', and 'goal'
        traj = self.wm.imagine_carry(
            policy, start, self.config.imag_horizon,
            self.initial(len(start['is_first'])))
        traj['reward_extr'] = self.extr_reward(traj)
        traj['reward_expl'] = self.expl_reward(traj)
        traj['reward_goal'] = self.goal_reward(traj)
        traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)

        def worker_loss(traj):
            wtraj = self.split_traj(traj)
            worker_loss, worker_metrics = self.worker.loss(wtraj)
            worker_metrics = {"worker_" + k: v for k,
                              v in worker_metrics.items()}
            return train_worker.astype(jnp.float32) * worker_loss, (traj, worker_metrics, wtraj)

        def manager_loss(traj):
            mtraj = self.abstract_traj(traj)
            manager_loss, manager_metrics = self.manager.loss(mtraj)
            manager_metrics = {"manager_" + k: v for k,
                               v in manager_metrics.items()}
            return train_manager.astype(jnp.float32) * manager_loss, (traj, manager_metrics, mtraj)

        ########################################################################
        #
        # WORKER TRAINING
        #
        ########################################################################

        # update the worker actor
        # lambda dummy_opt: actor, loss, traj, has_aux = 
        wmets, (traj, worker_metrics, wtraj) = self.worker.opt(
            self.worker.actor, worker_loss, traj, has_aux=True)
        worker_metrics.update(wmets)
        # update the worker critic
        for key, critic in self.worker.critics.items():
            cwmets = critic.train(wtraj, self.worker.actor)
            worker_metrics.update(
                {f'{key}_worker-critic_{k}': v for k, v in cwmets.items()})
        worker_metrics['worker_actor_opt_loss'] = worker_metrics['actor_opt_loss']
        


        ########################################################################
        #
        # MANAGER TRAINING
        #
        ########################################################################
        # update the manager actor
        mmets, (traj, manager_metrics, mtraj) = self.manager.opt(
            self.manager.actor, manager_loss, traj, has_aux=True)
        manager_metrics.update(mmets)
        # update the manager critic
        for key, critic in self.manager.critics.items():
            mmets = critic.train(mtraj, self.manager.actor)
            manager_metrics.update(
                {f'{key}_manager-critic_{k}': v for k, v in mmets.items()})
        manager_metrics['manager_actor_opt_loss'] = manager_metrics['actor_opt_loss']

        return traj, {**worker_metrics, **manager_metrics}

    def train_jointly0(self, imagine, start, train_manager=True, train_worker=True):
        start = start.copy()
        policy = functools.partial(self.policy, imag=True)
        traj = self.wm.imagine_carry(
            policy, start, self.config.imag_horizon,
            self.initial(len(start['is_first'])))
        traj['reward_extr'] = self.extr_reward(traj)
        traj['reward_expl'] = self.expl_reward(traj)
        traj['reward_goal'] = self.goal_reward(traj)
        traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)

        worker_metrics = {}
        manager_metrics = {}

        worker_metrics_keys = ['worker_actor_opt_loss'] + [f'{key}_worker-critic_{k}' for key in self.worker.critics.keys() for k in ['loss', 'grad_norm']]
        manager_metrics_keys = ['manager_actor_opt_loss'] + [f'{key}_manager-critic_{k}' for key in self.manager.critics.keys() for k in ['loss', 'grad_norm']]

        no_op_worker_fn = lambda traj: {key: 0.0 for key in worker_metrics_keys}
        no_op_manager_fn = lambda traj: {key: 0.0 for key in manager_metrics_keys}

        def train_worker_fn(traj):
            def worker_loss(traj):
                wtraj = self.split_traj(traj)
                worker_loss, worker_metrics = self.worker.loss(wtraj)
                worker_metrics = {"worker_" + k: v for k, v in worker_metrics.items()}
                return worker_loss, (traj, worker_metrics, wtraj)

            wmets, (traj, worker_metrics, wtraj) = self.worker.opt(
                self.worker.actor, worker_loss, traj, has_aux=True)
            worker_metrics.update(wmets)

            for key, critic in self.worker.critics.items():
                cwmets = critic.train(wtraj, self.worker.actor)
                worker_metrics.update(
                    {f'{key}_worker-critic_{k}': v for k, v in cwmets.items()})
            worker_metrics['worker_actor_opt_loss'] = worker_metrics['actor_opt_loss']
            return worker_metrics

        worker_metrics = jax.lax.cond(train_worker, lambda _: train_worker_fn(traj), lambda _: no_op_worker_fn(traj), operand=None)

        def train_manager_fn(traj):
            def manager_loss(traj):
                mtraj = self.abstract_traj(traj)
                manager_loss, manager_metrics = self.manager.loss(mtraj)
                manager_metrics = {"manager_" + k: v for k, v in manager_metrics.items()}
                return manager_loss, (traj, manager_metrics, mtraj)

            mmets, (traj, manager_metrics, mtraj) = self.manager.opt(
                self.manager.actor, manager_loss, traj, has_aux=True)
            manager_metrics.update(mmets)

            for key, critic in self.manager.critics.items():
                mmets = critic.train(mtraj, self.manager.actor)
                manager_metrics.update(
                    {f'{key}_manager-critic_{k}': v for k, v in mmets.items()})
            manager_metrics['manager_actor_opt_loss'] = manager_metrics['actor_opt_loss']
            return manager_metrics

        manager_metrics = jax.lax.cond(train_manager, lambda _: train_manager_fn(traj), lambda _: no_op_manager_fn(traj), operand=None)

        return traj, {**worker_metrics, **manager_metrics}


    def train_jointly_old(self, imagine, start):
        # start = start.copy()
        # metrics = {}
        # sg = lambda x: jnp.nest.map_structure(jaxutils.sg, x)
        # context = self.feat(start)
        # with jnp.GradientTape(persistent=True) as tape:
        #   skill = self.manager.actor(sg(start)).sample()
        #   goal = self.dec({'skill': skill, 'context': context}).mode()
        #   goal = (
        #       self.feat(start).astype(jnp.float32) + goal
        #       if self.config.manager_delta else goal)
        #   worker = lambda s: self.worker.actor(sg({
        #       **s, 'goal': goal, 'delta': goal - self.feat(s)})).sample()
        #   traj = imagine(worker, start, self.config.imag_horizon)
        #   traj['goal'] = jnp.repeat(goal[None], 1 + self.config.imag_horizon, 0)
        #   traj['skill'] = jnp.repeat(skill[None], 1 + self.config.imag_horizon, 0)
        #   traj['reward_extr'] = self.extr_reward(traj)
        #   traj['reward_expl'] = self.expl_reward(traj)
        #   traj['reward_goal'] = self.goal_reward(traj)
        #   traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)
        #   wtraj = traj.copy()
        #   mtraj = self.abstract_traj_old(traj)
        # mets = self.worker.update(wtraj, tape)
        # metrics.update({f'worker_{k}': v for k, v in mets.items()})
        # mets = self.manager.update(mtraj, tape)
        # metrics.update({f'manager_{k}': v for k, v in mets.items()})
        # return traj, metrics
        raise NotImplementedError

    def train_manager(self, imagine, start):
        # start = start.copy()
        # # with jnp.GradientTape(persistent=True) as tape:
        # policy = functools.partial(self.policy, imag=True)
        # traj = self.wm.imagine_carry(
        #     policy, start, self.config.imag_horizon,
        #     self.initial(len(start['is_first'])))
        # traj['reward_extr'] = self.extr_reward(traj)
        # traj['reward_expl'] = self.expl_reward(traj)
        # traj['reward_goal'] = self.goal_reward(traj)
        # traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)
        # mtraj = self.abstract_traj(traj)
        # metrics = self.manager.update(mtraj)  # , tape)
        # metrics = {f'manager_{k}': v for k, v in metrics.items()}

        start = start.copy()
        policy = functools.partial(self.policy, imag=True)
        # carry is an HRL dict containing 'step', 'skill', and 'goal'
        traj = self.wm.imagine_carry(
            policy, start, self.config.imag_horizon,
            self.initial(len(start['is_first'])))
        traj['reward_extr'] = self.extr_reward(traj)
        traj['reward_expl'] = self.expl_reward(traj)
        traj['reward_goal'] = self.goal_reward(traj)
        traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)

        def manager_loss(traj):
            mtraj = self.abstract_traj(traj)
            manager_loss, manager_metrics = self.manager.loss(mtraj)
            manager_metrics = {"manager_" + k: v for k,
                               v in manager_metrics.items()}
            return manager_loss, (traj, manager_metrics, mtraj)


        # update the manager actor
        mmets, (traj, manager_metrics, mtraj) = self.manager.opt(
            self.manager.actor, manager_loss, traj, has_aux=True)
        manager_metrics.update(mmets)
        # update the manager critic
        for key, critic in self.manager.critics.items():
            mmets = critic.train(mtraj, self.manager.actor)
            manager_metrics.update(
                {f'{key}_manager-critic_{k}': v for k, v in mmets.items()})
        manager_metrics['manager_actor_opt_loss'] = manager_metrics['actor_opt_loss']
        return traj, manager_metrics

    def train_worker(self, imagine, start, goal):
        # start = start.copy()
        # metrics = {}
        def worker(s): return self.worker.actor(jaxutils.sg({
            **s, 'goal': goal, 'delta': goal - self.feat(s).astype(jnp.float32),
        })).sample(seed=nj.rng())
        # traj = imagine(worker, start, self.config.imag_horizon)
        # traj['goal'] = jnp.repeat(goal[None], 1 + self.config.imag_horizon, 0)
        # traj['reward_extr'] = self.extr_reward(traj)
        # traj['reward_expl'] = self.expl_reward(traj)
        # traj['reward_goal'] = self.goal_reward(traj)
        # traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)
        # mets = self.worker.update(traj)  # , tape)
        # metrics.update({f'worker_{k}': v for k, v in mets.items()})
        traj = self.wm.imagine_carry(
            worker, start, self.config.imag_horizon,
            self.initial(len(start['is_first'])))
        traj['reward_extr'] = self.extr_reward(traj)
        traj['reward_expl'] = self.expl_reward(traj)
        traj['reward_goal'] = self.goal_reward(traj)
        traj['delta'] = traj['goal'] - self.feat(traj).astype(jnp.float32)

        def worker_loss(traj):
            wtraj = self.split_traj(traj)
            worker_loss, worker_metrics = self.worker.loss(wtraj)
            worker_metrics = {"worker_" + k: v for k,
                              v in worker_metrics.items()}
            return train_worker.astype(jnp.float32) * worker_loss, (traj, worker_metrics, wtraj)

        
        # update the worker actor
        wmets, (traj, worker_metrics, wtraj) = self.worker.opt(
            self.worker.actor, worker_loss, traj, has_aux=True)
        worker_metrics.update(wmets)
        # update the worker critic
        for key, critic in self.worker.critics.items():
            cwmets = critic.train(wtraj, self.worker.actor)
            worker_metrics.update(
                {f'{key}_worker-critic_{k}': v for k, v in cwmets.items()})
        worker_metrics['worker_actor_opt_loss'] = worker_metrics['actor_opt_loss']

        return traj, metrics

    def train_state_abstraction(self, data):
        metrics = {}
        state = context = self.abstract_state_feat(data)
        enc = self.state_abstraction_encoder({'state': state, 'context': context})
        dec = self.state_abstraction_decoder(
            {'abstract_state': enc.sample(seed=nj.rng()), 'context': context})

        rec = -dec.log_prob(jaxutils.sg(state))

        if self.config.use_state_abstraction_kl:
            kl = jaxutils.tfd.kl_divergence(enc, self.state_abstraction_prior)
            kl, mets = self.state_abstraction_kl(kl)
            metrics.update({f'state_abstract_kl_{k}': v for k, v in mets.items()})
            assert rec.shape == kl.shape, (rec.shape, kl.shape)
        else:
            kl = 0.0

        def loss_fn(rec, kl): return (rec + kl).mean()
        metrics.update(self.state_abstraction_kl_opt(
            [self.state_abstraction_encoder, self.state_abstraction_decoder], loss_fn, rec, kl))
        metrics['state_abstraction_rec_mean'] = rec.mean()
        metrics['state_abstraction_rec_std'] = rec.std()

        return metrics

    def train_vae_replay(self, data):
        metrics = {}
        feat = self.feat(data).astype(jnp.float32)
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
            enc = self.enc({'goal': goal, 'context': context})
            dec = self.dec({'skill': enc.sample(seed=nj.rng()), 'context': context})
            rec = -dec.log_prob(jaxutils.sg(goal))
            if self.config.goal_kl:
                kl = jaxutils.tfd.kl_divergence(enc, self.prior)
                kl, mets = self.kl(kl)
                local_metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
                assert rec.shape == kl.shape, (rec.shape, kl.shape)
            else:
                kl = 0.0

            local_metrics['goalrec_mean'] = rec.mean()
            local_metrics['goalrec_std'] = rec.std()
            
            return (rec + kl).mean(), local_metrics

        # def loss_fn(rec, kl): return (rec + kl).mean()
        # metrics.update(self.opt([self.enc, self.dec], loss_fn, rec, kl))
        mets, local_metrics = self.opt([self.enc, self.dec], loss_fn, goal, context, has_aux=True)
        # metrics.update(self.opt([self.enc, self.dec], loss_fn, goal, context, has_aux=True))

        # jax.debug.print('Grad Norm : {}', mets['autoenc_opt_grad_norm'])
        # metrics.update(self.opt(loss, [self.enc, self.dec]))
        # metrics['goalrec_mean'] = rec.mean()
        # metrics['goalrec_std'] = rec.std()
        metrics.update(mets)
        metrics.update(local_metrics)

        return metrics

    def train_vae_imag(self, traj):
        # metrics = {}
        # feat = self.feat(traj).astype(jnp.float32)
        # if 'context' in self.config.goal_decoder.inputs:
        #   if self.config.vae_span:
        #     context = feat[0]
        #     goal = feat[-1]
        #   else:
        #     assert feat.shape[0] > self.config.train_skill_duration
        #     context = feat[:-self.config.train_skill_duration]
        #     goal = feat[self.config.train_skill_duration:]
        # else:
        #   goal = context = feat
        # with jnp.GradientTape() as tape:
        #   enc = self.enc({'goal': goal, 'context': context})
        #   dec = self.dec({'skill': enc.sample(), 'context': context})
        #   rec = -dec.log_prob(jaxutils.sg(goal.astype(jnp.float32)))
        #   if self.config.goal_kl:
        #     kl = jaxutils.tfd.kl_divergence(enc, self.prior)
        #     kl, mets = self.kl(kl)
        #     metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
        #   else:
        #     kl = 0.0
        #   loss = (rec + kl).mean()
        # metrics.update(self.opt(tape, loss, [self.enc, self.dec]))
        # metrics['goalrec_mean'] = rec.mean()
        # metrics['goalrec_std'] = rec.std()
        # return metrics
        raise NotImplementedError

    def propose_goal(self, start, impl):
        feat = self.feat(start).astype(jnp.float32)
        if impl == 'replay':
            target = jax.random.shuffle(nj.rng(), feat).astype(jnp.float32)
            skill = self.enc({'goal': target, 'context': feat}
                             ).sample(seed=nj.rng())
            return self.dec({'skill': skill, 'context': feat}).mode()
        if impl == 'replay_direct':
            return jax.random.shuffle(nj.rng(), feat).astype(jnp.float32)
        if impl == 'manager':    
            # TODO(ag): This needs to be less hard-coded    
            manager_inputs = copy.deepcopy(start)
            if self.config.use_state_abstraction:    
              state = context = self.abstract_state_feat(start)
              state_abstraction_dist = self.state_abstraction_encoder({'state': state, 'context': context})
              abstract_state = state_abstraction_dist.sample(seed=nj.rng())
              manager_inputs['abstract_state'] = abstract_state
          
            skill = self.manager.actor(manager_inputs).sample(seed=nj.rng())
            goal = self.dec({'skill': skill, 'context': feat}).mode()
            goal = feat + goal if self.config.manager_delta else goal
            return goal
        if impl == 'prior':
            skill = self.prior.sample(len(start['is_terminal']), seed=nj.rng())
            return self.dec({'skill': skill, 'context': feat}).mode()
        raise NotImplementedError(impl)

    def goal_reward(self, traj):
        feat = self.feat(traj).astype(jnp.float32)
        goal = jaxutils.sg(traj['goal'].astype(jnp.float32))
        skill = jaxutils.sg(traj['skill'].astype(jnp.float32))
        context = jaxutils.sg(
            jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0))
        if self.config.goal_reward == 'dot':
            return jnp.einsum('...i,...i->...', goal, feat)[1:]
        elif self.config.goal_reward == 'dir':
            return jnp.einsum(
                '...i,...i->...', jnp.nn.l2_normalize(goal, -1), feat)[1:]
        elif self.config.goal_reward == 'normed_inner':
            norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
            return jnp.einsum('...i,...i->...', goal / norm, feat / norm)[1:]
        elif self.config.goal_reward == 'normed_squared':
            norm = jnp.linalg.norm(goal, axis=-1, keepdims=True)
            return -((goal / norm - feat / norm) ** 2).mean(-1)[1:]
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
            goal = jnp.nn.l2_normalize(goal[:-1], -1)
            diff = jnp.concatenate([feat[1:] - feat[:-1]], 0)
            return jnp.einsum('...i,...i->...', goal, diff)
        elif self.config.goal_reward == 'norm':
            return -jnp.linalg.norm(goal - feat, axis=-1)[1:]
        elif self.config.goal_reward == 'squared':
            return -((goal - feat) ** 2).sum(-1)[1:]
        elif self.config.goal_reward == 'epsilon':
            return ((goal - feat).mean(-1) < 1e-3).astype(jnp.float32)[1:]
        elif self.config.goal_reward == 'enclogprob':
            return self.enc({'goal': goal, 'context': context}).log_prob(skill)[1:]
        elif self.config.goal_reward == 'encprob':
            return self.enc({'goal': goal, 'context': context}).prob(skill)[1:]
        elif self.config.goal_reward == 'enc_normed_cos':
            dist = self.enc({'goal': goal, 'context': context})
            probs = dist.distribution.probs_parameter()
            norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return jnp.einsum('...ij,...ij->...', probs / norm, skill / norm)[1:]
        elif self.config.goal_reward == 'enc_normed_squared':
            dist = self.enc({'goal': goal, 'context': context})
            probs = dist.distribution.probs_parameter()
            norm = jnp.linalg.norm(probs, axis=[-2, -1], keepdims=True)
            return -((probs / norm - skill / norm) ** 2).mean([-2, -1])[1:]
        else:
            raise NotImplementedError(self.config.goal_reward)

    def elbo_reward(self, traj):
        feat = self.feat(traj).astype(jnp.float32)
        context = jnp.repeat(feat[0][None], 1 + self.config.imag_horizon, 0)
        enc = self.enc({'goal': feat, 'context': context})
        dec = self.dec({'skill': enc.sample(
            seed=nj.rng()), 'context': context})
        if self.config.adver_impl == 'abs':
            return jnp.abs(dec.mode() - feat).mean(-1)[1:]
        elif self.config.adver_impl == 'squared':
            return ((dec.mode() - feat) ** 2).mean(-1)[1:]
        ll = dec.log_prob(feat)
        kl = jaxutils.tfd.kl_divergence(enc, self.prior)
        if self.config.adver_impl == 'elbo_scaled':
            return (kl - ll / self.kl.scale())[1:]
        elif self.config.adver_impl == 'elbo_unscaled':
            return (kl - ll)[1:]
        raise NotImplementedError(self.config.adver_impl)

    def split_traj(self, traj):
        traj = traj.copy()
        k = self.config.train_skill_duration
        # Q: Why do we need this assert ?
        assert len(traj['action']) % k == 1
        def reshape(x): return x.reshape(
            [x.shape[0] // k, k] + list(x.shape[1:]))
        for key, val in list(traj.items()):
            val = jnp.concatenate([0 * val[:1], val],
                                  0) if 'reward' in key else val
            # (1 2 3 4 5 6 7 8 9 10) -> ((1 2 3 4) (4 5 6 7) (7 8 9 10))
            val = jnp.concatenate([reshape(val[:-1]), val[k::k][:, None]], 1)
            # N val K val B val F... -> K val (N B) val F...
            val = val.transpose([1, 0] + list(range(2, len(val.shape))))
            val = val.reshape(
                [val.shape[0], np.prod(val.shape[1:3])] + list(val.shape[3:]))
            val = val[1:] if 'reward' in key else val
            traj[key] = val
        # Bootstrap sub trajectory against current not next goal.
        traj['goal'] = jnp.concatenate(
            [traj['goal'][:-1], traj['goal'][:1]], 0)
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
        def reshape(x): return x.reshape(
            [x.shape[0] // k, k] + list(x.shape[1:]))
        weights = jnp.cumprod(reshape(traj['cont'][:-1]), 1)
        for key, value in list(traj.items()):
            if 'reward' in key:
                traj[key] = (reshape(value) * weights).mean(1)
            elif key == 'cont':
                traj[key] = jnp.concatenate(
                    [value[:1], reshape(value[1:]).prod(1)], 0)
            else:
                traj[key] = jnp.concatenate(
                    [reshape(value[:-1])[:, 0], value[-1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
        return traj

    def abstract_traj_old(self, traj):
        # traj = traj.copy()
        # traj['action'] = traj.pop('skill')
        # mult = jnp.math.cumprod(traj['cont'][1:], 0)
        # for key, value in list(traj.items()):
        #   if 'reward' in key:
        #     traj[key] = (mult * value).mean(0)[None]
        #   elif key == 'cont':
        #     traj[key] = jnp.stack([value[0], value[1:].prod(0)], 0)
        #   else:
        #     traj[key] = jnp.stack([value[0], value[-1]], 0)
        # return traj
        raise NotImplementedError

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
        if self.short_circuit_worker:
            return {}
        # Prepare initial state.
        # data['observation] is (batch_size, batch_length, observation_dim)
        decoder = self.wm.heads['decoder']
        # take first 6 samples from the batch (Why?)
        states, _ = self.wm.rssm.observe(
            self.wm.encoder(data)[:6], data['action'][:6], data['is_first'][:6])
        start = {k: v[:, 4] for k, v in states.items()}
        start['is_terminal'] = data['is_terminal'][:6, 4]
        goal = self.propose_goal(start, impl)
        # goal is (6, batch_length, state_dim)
        # self.feat(states) is the same shape as goal
        # Worker rollout.

        def worker(s): return self.worker.actor({
            **s, 'goal': goal, 'delta': goal - self.feat(s).astype(jnp.float32),
        }).sample(seed=nj.rng())
        traj = self.wm.imagine(
            worker, start, self.config.worker_report_horizon)
        # Decoder into images.
        initial = decoder(start)
        target = decoder(
            {'deter': goal, 'stoch': self.wm.rssm.get_stoch(goal)})
        rollout = decoder(traj)
        # Stich together into videos. Currently, rollout['observation'] is a 
        # SymLogDist; have mean(), mode(); they're [B, T, D] with the tabular envs
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
                t_central = frame_extracts[:, :, 0, :4 :]
                t_left = frame_extracts[:, :, 0, 4:6, :]
                t_right = frame_extracts[:, :, 0, 6:8, :]
                frame = -jnp.ones((B, T, 6, 7, 1))
                # update blank_frame with the goal representation
                frame = frame.at[:, :, 1:5, 3, :].set(t_central)
                frame = frame.at[:, :, 1, 1:3, :].set(t_left)
                frame = frame.at[:, :, 1, 4:6, :].set(t_right)
                videos[k] = jaxutils.video_grid(frame)

        return videos
