import copy

from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from envs import ModifiedFrozenLake
import torch as th
from torch.nn import functional as F
import numpy as np
from typing import Optional, Tuple

import gym

from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation, safe_mean

from stable_baselines3.dqn import DQN


def f_min(qs:[th.Tensor]):
    """use on AND composition. Is expected to yield an upper bound for the Q~*"""
    return th.min(th.stack(qs, dim=1), dim=1)[0]
def upper_bound_violation(q, bounds):
    return th.clip(q - bounds, min=0)
def bound_upper(q, bounds):
    return th.clip(q, max=bounds)

def f_max(qs):
    """use on OR composition. Is expected to yield a lower bound for the Q~*"""
    return th.max(th.stack(qs, dim=1), dim=1)[0]
def lower_bound_violation(q, bounds):
    return th.clip(q - bounds, max=0)
def bound_lower(q, bounds):
    return th.clip(q, min=bounds)

def f_mean(qs):
    return th.mean(th.stack([th.cat(q, dim=1) for q in qs], dim=2), dim=2)[0]

fs = {
    'min': f_min,
    'max': f_max,
    'mean': f_mean
}

comp_to_f = {
    'and': f_min,
    'or': f_max
}
comp_to_bound = {
    'and': bound_upper,
    'or': bound_lower
}
comp_to_bound_violation = {
    'and': upper_bound_violation,
    'or': lower_bound_violation
}


class CompBoundedDQN(DQN):
    def __init__(self, models: [DQN], comp_type: str, clip_method:str, *args, **kwargs):
        """
        :param models: list of pretrained DQN models for subtasks
        :param comp_type: 'and' or 'or'
        :param clip_method: 'soft', 'hard', or 'test'
        """
        assert comp_type in comp_to_f.keys(), f'comp_type must be one of {comp_to_f.keys()}'
        super(CompBoundedDQN, self).__init__(*args, **kwargs)
        self.pretrained_models = models
        self.fcomp = comp_to_f[comp_type]
        self.bound_violation = comp_to_bound_violation[comp_type]
        self.fbound = comp_to_bound[comp_type]
        self.clip_method = clip_method
        self.cummulative_mean_rollout_reward = 0
        self.to('cpu')

    def to(self, device):
        self.q_net.to(device)
        self.q_net_target.to(device)
        for model in self.pretrained_models:
            model.q_net.to(device)
            model.q_net_target.to(device)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        self.to(self.device)

        losses = []
        clip_losses = []
        bound_violations = []
        self.cummulative_mean_rollout_reward += safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        self.logger.record("rollout/tot_ep_rew_mean", self.cummulative_mean_rollout_reward)
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # Compute bounds next Q values of pretrained models
                target_q_values_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    next_q_values_ = m.q_net(replay_data.next_observations)
                    next_q_values_, _ = th.max(next_q_values_, dim=1)
                    target_q_values_pretrained.append(next_q_values_)
                # calculate bounds based on the pretrained models
                target_q_bounds = self.fcomp(target_q_values_pretrained).unsqueeze(1)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            clip_loss = th.tensor([0.0]).to(self.device)
            if self.clip_method == 'soft':
                # calculate soft update using the current q model
                next_q_values = self.q_net(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                violations = self.bound_violation(target_q_values_current, target_q_bounds)
                clip_loss += F.smooth_l1_loss(violations, th.zeros_like(violations))
                bound_violations.append(violations.sum().item())
            elif self.clip_method == 'hard':
                with th.no_grad():
                    # count the number of violations
                    next_q_values = self.q_net(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    violations = self.bound_violation(target_q_values_current, target_q_bounds).sum().item()
                    bound_violations.append(violations)
                target_q_values = self.fbound(target_q_values, target_q_bounds)
            elif self.clip_method == 'soft_hard':
                # calculate soft update using the current q model
                next_q_values = self.q_net(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                violations = self.bound_violation(target_q_values_current, target_q_bounds)
                clip_loss += F.smooth_l1_loss(violations, th.zeros_like(violations))
                bound_violations.append(violations.sum().item())
                target_q_values = self.fbound(target_q_values, target_q_bounds)
            elif self.clip_method == 'none' or self.clip_method == 'test':
                # only save the number of violations
                with th.no_grad():
                    next_q_values = self.q_net(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    violations = self.bound_violation(target_q_values_current, target_q_bounds).sum().item()
                    bound_violations.append(violations)

            clip_losses.append(clip_loss.item())
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values) + clip_loss
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/clip_loss", np.mean(clip_losses))
        self.logger.record('train/bound_violations', np.mean(bound_violations))
        self.to('cpu')

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            # Apply clipping here at the current state
            if self.clip_method == 'test':
                # Compute bounds next Q values of pretrained models
                q_values_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    obs, _ = m.policy.obs_to_tensor(observation)
                    q_values_ = m.q_net(obs)
                    q_values_pretrained.append(q_values_)
                # calculate bounds based on the pretrained models
                q_bounds = self.fcomp(q_values_pretrained)

                # Get current Q-values estimates
                current_q_values = self.q_net(th.tensor(observation))
                # bound the current q values
                current_q_values = self.fbound(current_q_values, q_bounds)
                # Follow greedy policy: use the one with the highest value
                current_q_values, action = current_q_values.max(dim=-1)
                action = action.numpy()
            elif self.clip_method == 'infer':
                # Compute bounds next Q values of pretrained models
                q_values_pretrained = []
                actions_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    q_values_ = m.q_net(th.tensor(observation))
                    q_values_, act_ = th.max(q_values_, dim=1)
                    q_values_pretrained.append(q_values_)
                    actions_pretrained.append(act_)
                # calculate bounds based on the pretrained models
                q_bounds = self.fcomp(q_values_pretrained).unsqueeze(1)
                # select actions, which values were used by the composition function
                t_qvp = th.stack(q_values_pretrained, dim=0)
                idx = (t_qvp == q_bounds).nonzero(as_tuple=True)[0]
                action = actions_pretrained[idx].numpy()
            else:
                action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state


class CompDQNNet(QNetwork):
    def __init__(self,
                 models=None,
                 comp_f=None,
                 action_space=None,
                 observation_space=None,
                 normalize_images=False):
        super(CompDQNNet, self).__init__(
            features_extractor=None,
            observation_space=observation_space,
            action_space=action_space,
            features_dim=1,
            net_arch=[1],
            activation_fn=th.nn.ReLU,
            normalize_images=normalize_images,
        )
        self.models = copy.deepcopy(models)
        self.comp_f = comp_f
        self.q_net = th.nn.Sequential(
            th.nn.ModuleList([ m.q_net for m in self.models ]),
        )

    def forward(self, obs):
        q_values = []
        for m in self.models:
            q_values.append(m.q_net(obs))
        return self.comp_f(q_values)

    def _predict(self, observation: th.Tensor, deterministic: bool = True):
        # Compute bounds next Q values of pretrained models
        q_values_pretrained = []
        actions_pretrained = []
        for m in self.models:
            # Compute the next Q values: min over all critics targets
            q_values_ = m.q_net(th.tensor(observation))
            q_values_, act_ = th.max(q_values_, dim=1)
            q_values_pretrained.append(q_values_)
            actions_pretrained.append(act_)
        # calculate bounds based on the pretrained models
        q_bounds = self.comp_f(q_values_pretrained).unsqueeze(1)
        # select actions, which values were used by the composition function
        t_qvp = th.stack(q_values_pretrained, dim=0)
        idx = (t_qvp == q_bounds).nonzero(as_tuple=True)[0]
        action = actions_pretrained[idx]
        return action


class CompDQNPolicy(DQNPolicy):
    def __init__(
            self, observation_space,action_space,lr_schedule,
            models=None, comp_f=None, **kwargs
    ):
        self.pmodels = models
        self.comp_f = comp_f
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr_schedule = lr_schedule
        super(CompDQNPolicy, self).__init__(observation_space,action_space,lr_schedule, **kwargs)
    def make_q_net(self):
        return CompDQNNet(
            action_space=self.action_space,
            observation_space=self.observation_space,
            models=self.pmodels,
            comp_f=self.comp_f)


class WarmCompBoundedDQN(DQN):
    def __init__(self, env, models: [DQN], comp_type: str, clip_method:str, **kwargs):
        """
        Warm-started Composed Bounded DQN
        :param models: list of pretrained DQN models for subtasks
        :param comp_type: 'and' or 'or'
        :param clip_method: 'soft', 'hard', or 'test'
        """
        assert comp_type in comp_to_f.keys(), f'comp_type must be one of {comp_to_f.keys()}'
        super(WarmCompBoundedDQN, self).__init__(
            CompDQNPolicy,
            env,
            policy_kwargs={
                "models":models,
                "comp_f":comp_to_f[comp_type],
            },
            **kwargs)
        self.pretrained_models = models
        self.fcomp = comp_to_f[comp_type]
        self.bound_violation = comp_to_bound_violation[comp_type]
        self.fbound = comp_to_bound[comp_type]
        self.clip_method = clip_method
        self.cummulative_mean_rollout_reward = 0
        self.to('cpu')

    def to(self, device):
        self.q_net.to(device)
        self.q_net_target.to(device)
        for model in self.pretrained_models:
            model.q_net.to(device)
            model.q_net_target.to(device)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        self.to(self.device)

        losses = []
        clip_losses = []
        bound_violations = []
        self.cummulative_mean_rollout_reward += safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        self.logger.record("rollout/tot_ep_rew_mean", self.cummulative_mean_rollout_reward)
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                # Compute bounds next Q values of pretrained models
                target_q_values_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    next_q_values_ = m.q_net(replay_data.next_observations)
                    next_q_values_, _ = th.max(next_q_values_, dim=1)
                    target_q_values_pretrained.append(next_q_values_)
                # calculate bounds based on the pretrained models
                target_q_bounds = self.fcomp(target_q_values_pretrained).unsqueeze(1)

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            clip_loss = th.tensor([0.0]).to(self.device)
            if self.clip_method == 'soft':
                # calculate soft update using the current q model
                next_q_values = self.q_net(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                violations = self.bound_violation(target_q_values_current, target_q_bounds)
                clip_loss += F.smooth_l1_loss(violations, th.zeros_like(violations))
                bound_violations.append(violations.sum().item())
            elif self.clip_method == 'hard':
                with th.no_grad():
                    # count the number of violations
                    next_q_values = self.q_net(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    violations = self.bound_violation(target_q_values_current, target_q_bounds).sum().item()
                    bound_violations.append(violations)
                target_q_values = self.fbound(target_q_values, target_q_bounds)
            elif self.clip_method == 'soft_hard':
                # calculate soft update using the current q model
                next_q_values = self.q_net(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                violations = self.bound_violation(target_q_values_current, target_q_bounds)
                clip_loss += F.smooth_l1_loss(violations, th.zeros_like(violations))
                bound_violations.append(violations.sum().item())
                target_q_values = self.fbound(target_q_values, target_q_bounds)
            elif self.clip_method == 'none' or self.clip_method == 'test':
                # only save the number of violations
                with th.no_grad():
                    next_q_values = self.q_net(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values_current = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                    violations = self.bound_violation(target_q_values_current, target_q_bounds).sum().item()
                    bound_violations.append(violations)

            clip_losses.append(clip_loss.item())
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values) + clip_loss
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/clip_loss", np.mean(clip_losses))
        self.logger.record('train/bound_violations', np.mean(bound_violations))
        self.to('cpu')

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            # Apply clipping here at the current state
            if self.clip_method == 'test':
                # Compute bounds next Q values of pretrained models
                q_values_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    obs, _ = m.policy.obs_to_tensor(observation)
                    q_values_ = m.q_net(obs)
                    q_values_pretrained.append(q_values_)
                # calculate bounds based on the pretrained models
                q_bounds = self.fcomp(q_values_pretrained)

                # Get current Q-values estimates
                current_q_values = self.q_net(th.tensor(observation))
                # bound the current q values
                current_q_values = self.fbound(current_q_values, q_bounds)
                # Follow greedy policy: use the one with the highest value
                current_q_values, action = current_q_values.max(dim=-1)
                action = action.numpy()
            elif self.clip_method == 'infer':
                # Compute bounds next Q values of pretrained models
                q_values_pretrained = []
                actions_pretrained = []
                for m in self.pretrained_models:
                    # Compute the next Q values: min over all critics targets
                    q_values_ = m.q_net(th.tensor(observation))
                    q_values_, act_ = th.max(q_values_, dim=1)
                    q_values_pretrained.append(q_values_)
                    actions_pretrained.append(act_)
                # calculate bounds based on the pretrained models
                q_bounds = self.fcomp(q_values_pretrained).unsqueeze(1)
                # select actions, which values were used by the composition function
                t_qvp = th.stack(q_values_pretrained, dim=0)
                idx = (t_qvp == q_bounds).nonzero(as_tuple=True)[0]
                action = actions_pretrained[idx].numpy()
            else:
                action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
