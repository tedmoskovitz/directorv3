import numpy as np
from copy import deepcopy
import dm_env
from dm_env import specs
import functools
import embodied
from typing import Optional, List, Tuple
import pdb


class FourRooms(dm_env.Environment):

  def __init__(
      self,
      start_state: int = 100,
      reset_goal: bool = False,
      lambda_: float = 1.0,
      seed: int = 0,
      goals: Optional[List[int]] = None,
      goal_rewards: Optional[List[float]] = None,
      observation_type: str = 'one-hot'):
    """Initializes the FourRooms environment.

    Args:
      start_state: The starting state of the agent.
      reset_goal: Whether to reset the goal each episode.
      lambda_: Decay rate for reward.
      seed: Random seed for reproducibility.
      discount: Discount factor for future rewards.
      goals: List of goal states.
      goal_rewards: List of rewards for reaching each goal.
      observation_type: Type of observation, either 'index' or 'one-hot'.
    """
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward
    W = -1
    G = 10
    np.random.seed(seed)
    self._W = W  # wall
    self._G = G  # goal
    self.lambda_ = lambda_

    self._layout = np.array([
        [W, W, W, W, W, W, W, W, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W],
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, W, W, W, W, W, 0, W, W, W],
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, 0, 0, 0, 0, 0, W], 
        [W, 0, 0, 0, 0, W, 0, 0, 0, 0, W], 
        [W, W, W, W, W, W, W, W, W, W, W], 
    ])
    # reward
    flat_layout = self._layout.flatten()
    wall_idxs = np.stack(np.where(flat_layout == W)).T
    # possible reward states are those where there isn't a wall
    self._available_states = np.array(
      [s for s in range(len(flat_layout)) if s not in wall_idxs])
    self._idx_layout = np.arange(self._layout.size).reshape(self._layout.shape)

    self._reset_goal = reset_goal
    self._random_start_state = start_state < 0
    if self._random_start_state:
      start_state_idx = np.random.choice(self._available_states)
    else:
      start_state_idx = start_state
    self._start_state = self.idx_to_state_coords(start_state_idx)
    self._episodes = 0
    self._state = self._start_state
    self._observation_type = observation_type
    self._start_obs = self.get_obs()
    self._number_of_states = np.prod(np.shape(self._layout))
    
    flat_layout = self._layout.flatten()
    self.wall_idxs = np.stack(np.where(flat_layout == W)).T
    # room layout:
    # 1 2
    # 0 3
    self._N = self._layout.shape[0]
    self.room0_idxs = list(self._idx_layout[self._N//2:, :self._N//2].flatten())
    self.room1_idxs = list(self._idx_layout[:self._N//2, :self._N//2].flatten())
    self.room2_idxs = list(self._idx_layout[:self._N//2, self._N//2:].flatten())
    self.room3_idxs = list(self._idx_layout[self._N//2:, self._N//2:].flatten())
    self.room0_idxs = [idx for idx in self.room0_idxs if idx not in self.wall_idxs]
    self.room1_idxs = [idx for idx in self.room1_idxs if idx not in self.wall_idxs]
    self.room2_idxs = [idx for idx in self.room2_idxs if idx not in self.wall_idxs]
    self.room3_idxs = [idx for idx in self.room3_idxs if idx not in self.wall_idxs]
    self.room_idxs = [self.room0_idxs, self.room1_idxs, self.room2_idxs, self.room3_idxs]
    
    self.r = deepcopy(flat_layout).astype(float)
    self.goal_rewards = np.array([G]) if goal_rewards is None else np.array(goal_rewards)
    if goals is None:
      goal_state = np.random.choice(self._available_states)
      self.goals = [goal_state]
      self.r[goal_state] = G
      self.goal_visits = {goal_state: 0}
    else:
      assert len(goals) == len(goal_rewards), 'Need same # of goals as rewards.'
      self.goal_visits = {}
      for i, goal in enumerate(goals):
        self.r[goal] = G if goal_rewards is None else goal_rewards[i]
        self.goal_visits[goal] = 0
      self.goals = goals
    
    self._goals_reached = 0
    self._steps = 0

    # transition matrix
    self._R = np.array([W, 0, G])
    P = np.zeros([self._number_of_states, 5, self._number_of_states])
    l = self._layout.shape[0]
    p = 1
    for a in range(5): 
      for s in range(self._number_of_states):
        for sp in range(self._number_of_states):
          
          if a == 0: 
            if sp == s - l and flat_layout[sp] != W: P[s, a, sp] =  p; 
            elif sp == s - l and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 1: 
            if sp == s + 1 and flat_layout[sp] != W: P[s, a, sp] = p; 
            elif sp == s + 1 and flat_layout[sp] == W: P[s, a, s] = p; 
          elif a == 2: 
            if sp == s + l and flat_layout[sp ] != W: P[s, a, sp] = p; 
            elif sp == s + l and flat_layout[sp] == W: P[s, a, s] = p;
          elif a == 3: 
            if sp == s - 1 and flat_layout[sp] != W: P[s, a, sp] = p;
            elif sp == s - 1 and flat_layout[sp] == W: P[s, a, s] = p;
          else:
            P[s, a, sp] = p if sp == s else 0

    self._P = P

  @property
  def number_of_states(self) -> int:
      return self._number_of_states

  @property
  def goal_states(self) -> List[int]:
      return self.goals

  def get_obs(self, state_coords: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Returns the current observation.

        Args:
            state_coords: Optional state coordinates. If not provided, uses the current state.

        Returns:
            An array representing the current observation.
        """
    if self._observation_type == 'index':
      row, col= self._state if state_coords is None else state_coords
      obs = row*self._layout.shape[1] + col
      return obs
    elif self._observation_type == 'one-hot':
      obs = np.zeros(self._layout.size)
      row, col= self._state if state_coords is None else state_coords
      obs[row*self._layout.shape[1] + col] = 1
      return obs
    else:
      raise ValueError('Observation type not supported')

  def idx_to_state_coords(self, obs: int) -> Tuple[int, int]:
    col = obs % self._layout.shape[1]
    row = obs // self._layout.shape[1]
    return (row, col)  # row, column

  def rewards_remaining(self) -> List[float]:
    """Calculates the remaining rewards in the environment.

        Returns:
            A list of remaining rewards for each goal.
    """
    return [self.goal_rewards[i] * self.lambda_ ** self.goal_visits[g] for i,g in enumerate(self.goals)]

  def get_r(self) -> np.ndarray:
    remaining_rewards = self.rewards_remaining()
    r = deepcopy(self.r).astype(float)
    for i, goal in enumerate(self.goals):
      r[goal] = remaining_rewards[i]
    return r
  
  @property
  def episodes(self):
      return self._episodes

  def P(self, s: int, a: int, sp: int, r: float) -> float:
    if r not in self._R: return 0; 
    r_idx = np.where(self._R == r)[0][0]
    return self._P[s, a, sp, r_idx]

  def reset(self) -> dm_env.TimeStep:
    """Resets the environment for a new episode.

    Returns:
      A dm_env.TimeStep representing the initial state of the new episode.
    """
    if self._random_start_state:
      start_state = np.random.choice(self._available_states)
      self._start_state = self.idx_to_state_coords(start_state)
    self._state = self._start_state
    self._steps = 0
    for goal in self.goals:
      self.goal_visits[goal] = 0
    return dm_env.restart(self.get_obs())

  def step(self, action: int) -> dm_env.TimeStep:
    """Takes a step in the environment using the given action.

    Args:
      action: The action to take in the environment.

    Returns:
      A dm_env.TimeStep representing the state of the environment after taking the action.
    """
    done = False
    row, col = self._state
    r2d = np.reshape(self.r, self._layout.shape)
    
    if action == 0:  # up
      new_state = (row - 1, col)
    elif action == 1:  # right
      new_state = (row, col+ 1)
    elif action == 2:  # down
      new_state = (row + 1, col)
    elif action == 3:  # left
      new_state = (row, col- 1)
    elif action == 4: # stay
      new_state = (row, col)
    else:
      raise ValueError(f"Invalid action: {action} is not 0, 1, 2, 3, or 4.")

    new_row, new_col = new_state
    reward = r2d[new_row, new_col]
    if self._layout[new_row, new_col] == self._W:  # wall
      new_state = (row, col)
    elif r2d[new_row, new_col] == self._G:  # a goal
      goal = new_row * self._layout.shape[1] + new_col
      reward = self.lambda_ ** self.goal_visits[goal] * r2d[new_row, new_col]
      self.goal_visits[goal] += 1
    # terminate if no reward left in the environment
    if np.max(self.rewards_remaining()) < 1e-1:
      done = True

    self._state = new_state
    self._steps += 1
    step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID
    discount = 0. if done else 1.0
    return dm_env.TimeStep(step_type, reward, discount, self.get_obs())
  
  def observation_spec(self):
    if self._observation_type == 'index':
      return specs.DiscreteArray(
        self._layout.size, dtype=np.int32, name='observation')
    elif self._observation_type == 'one-hot':
      return specs.Array(
        shape=(self._layout.size,), dtype=np.float32, name='observation')
    else:
      raise ValueError('Observation type not supported')
  
  def action_spec(self):
    return specs.DiscreteArray(num_values=5, dtype=np.int32, name='action')


class TMaze(dm_env.Environment):

  def __init__(
    self,
    reset_goal: bool = False,
    p_right: float = 1.0,  # 0.5
    max_steps: int = 12,
    observation_type: str = 'one-hot'):
    """Initializes the TMaze environment.

    Args:
      reset_goal: Whether to reset the goal each episode.
      p_right: Probability of choosing the right goal.
      max_steps: Maximum steps in an episode.
      discount: Discount factor for future rewards.
      observation_type: Type of observation, either 'index' or 'one-hot'.
    """
    # -1: wall
    # 0: empty, episode continues
    # other: number indicates reward, episode will terminate
    W = -1
    G = 10
    self._W = W  # wall
    self._G = G  # goal

    self._empty_layout = np.array([
        [W, W, W, W, W, W, W],
        [W, 0, 0, 0, 0, 0, W],
        [W, W, W, 0, W, W, W], 
        [W, W, W, 0, W, W, W], 
        [W, W, W, 0, W, W, W],  
        [W, W, W, W, W, W, W] 
    ])

    self._number_of_states = np.prod(np.shape(self._empty_layout))
    # reward
    flat_empty_layout = self._empty_layout.flatten()
    # possible reward states are those where there isn't a wall
    self._available_states = np.array([8, 12])
    self.r = deepcopy(flat_empty_layout)

    self.p_right = p_right
    goal_state = np.random.choice(
      self._available_states, p=[1 - self.p_right, self.p_right])
    self._goal_hist = [goal_state] 
    self.goal_state = goal_state
    self.r[goal_state] = G
    self._goals_reached = 0
    self._max_steps = max_steps
    self._reset_goal = reset_goal
    self._start_state = self.idx_to_state_coords(31)
    self._episodes = 0
    self._state = self._start_state
    self._steps = 0
    self._observation_type = observation_type
    self._start_obs = self.get_obs()

    # transition matrix
    self.R = np.array([W, 0, G])
    P = np.zeros([self._number_of_states, 4, self._number_of_states])
    l = self._empty_layout.shape[0]
    p = 1
    for a in range(4): 
      for s in range(self._number_of_states):
        for sp in range(self._number_of_states):
          
          if a == 0: 
            if sp == s - l and flat_empty_layout[sp] != W: P[s, a, sp] =  p; 
            elif sp == s - l and flat_empty_layout[sp] == W: P[s, a, s] = p; 
          elif a == 1: 
            if sp == s + 1 and flat_empty_layout[sp] != W: P[s, a, sp] = p; 
            elif sp == s + 1 and flat_empty_layout[sp] == W: P[s, a, s] = p; 
          elif a == 2: 
            if sp == s + l and flat_empty_layout[sp ] != W: P[s, a, sp] = p; 
            elif sp == s + l and flat_empty_layout[sp] == W: P[s, a, s] = p;
          else: 
            if sp == s - 1 and flat_empty_layout[sp] != W: P[s, a, sp] = p;
            elif sp == s - 1 and flat_empty_layout[sp] == W: P[s, a, s] = p;
      
    self._P = P
    

  @property
  def number_of_states(self) -> int:
      return self._number_of_states

  @property
  def goal_states(self) -> List[int]:
      return [8, 12]

  def get_obs(self, s: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Observation is state index and current goal index.

    Args:
        s (Tuple, optional): 2D state. Defaults to None.

    Returns:
        numpy array: [state index, current goal (-1 for left arm, +1 for right)] 
    """
    row, col = self._state if s is None else s
    # if showing goal every step OR if first step
    goal_feat = np.sign(self.goal_state - 10)  # -1 for left, +1 for right
    # w.p. goal_feature_noise flip goal feature
    if self._observation_type == 'index':
      return np.array([row*self._empty_layout.shape[1] + col, goal_feat])
    elif self._observation_type == 'one-hot':
      obs = np.zeros(self._number_of_states + 1)
      obs[row*self._empty_layout.shape[1] + col] = 1
      obs[-1] = goal_feat
      return obs

  def idx_to_state_coords(self, obs) -> Tuple[int, int]:
    col= obs % self._empty_layout.shape[1]
    row = obs // self._empty_layout.shape[1]
    return (row, col)

  @property
  def episodes(self) -> int:
      return self._episodes

  def P(self, s: int, a: int, sp: int, r: float) -> float:
    if r not in self._R: return 0; 
    r_idx = np.where(self._R == r)[0][0]
    return self._P[s, a, sp, r_idx]

  def reset(self) -> dm_env.TimeStep:
    self._state = self._start_state
    self._episodes = 0
    self.r = deepcopy(self._empty_layout.flatten())
    goal_state = np.random.choice(
      self._available_states, p=[1 - self.p_right, self.p_right])
    self._goal_hist.append(goal_state)
    self.goal_state = goal_state
    self.r[goal_state] = self._G
    self._steps = 0
    return dm_env.restart(self.get_obs())

  def step(self, action: int) -> dm_env.TimeStep:
    done = False
    row, col = self._state
    r2d = np.reshape(self.r, self._empty_layout.shape)
    
    if action == 0:  # up
      new_state = (row - 1, col)
    elif action == 1:  # right
      new_state = (row, col+ 1)
    elif action == 2:  # down
      new_state = (row + 1, col)
    elif action == 3:  # left
      new_state = (row, col- 1)
    else:
      raise ValueError("Invalid action: {} is not 0, 1, 2, or 3.".format(action))

    new_row, new_col = new_state
    reward = r2d[new_row, new_col]
    if r2d[new_row, new_col] == self._W:  # wall
      new_state = (row, col)
    elif r2d[new_row, new_col] == self._G:  # a goal
      self._episodes += 1
      done = True

    self._state = new_state
    self._steps += 1
    if self._steps >= self._max_steps:
        done = True

    step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID
    discount = 0.0 if done else 1.0
    return dm_env.TimeStep(step_type, reward, discount, self.get_obs())

  def observation_spec(self):
    if self._observation_type == 'index':
      return specs.Array((2,), dtype=np.int32, name='observation')
    elif self._observation_type == 'one-hot':
      return specs.Array(
        shape=(self._empty_layout.size + 1,), dtype=np.float32, name='observation')
    else:
      raise ValueError('Observation type not supported')
  
  def action_spec(self):
    return specs.DiscreteArray(num_values=4, dtype=np.int32, name='action')


NAME2ENV = dict(fourrooms=FourRooms, tmaze=TMaze,)

class TabularNavigationEnv(embodied.Env):

  def __init__(self, env, obs_key='observation', act_key='action'):
    env = NAME2ENV[env]()
    self._env = env
    obs_spec = self._env.observation_spec()
    act_spec = self._env.action_spec()
    self._obs_dict = isinstance(obs_spec, dict)
    self._act_dict = isinstance(act_spec, dict)
    self._obs_key = not self._obs_dict and obs_key
    self._act_key = not self._act_dict and act_key
    self._obs_empty = []
    self._done = True

  @functools.cached_property
  def obs_space(self):
    spec = self._env.observation_spec()
    spec = spec if self._obs_dict else {self._obs_key: spec}
    if 'reward' in spec:
      spec['obs_reward'] = spec.pop('reward')
    for key, value in spec.copy().items():
      if int(np.prod(value.shape)) == 0:
        self._obs_empty.append(key)
        del spec[key]
    return {
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        **{k or self._obs_key: self._convert(v) for k, v in spec.items()},
    }

  @functools.cached_property
  def act_space(self):
    spec = self._env.action_spec()
    spec = spec if self._act_dict else {self._act_key: spec}
    return {
        'reset': embodied.Space(bool),
        **{k or self._act_key: self._convert(v) for k, v in spec.items()},
    }

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    if reset or self._done:
      time_step = self._env.reset()
    else:
      action = action if self._act_dict else action[self._act_key]
      time_step = self._env.step(action)
    self._done = time_step.last()
    return self._obs(time_step)

  def _obs(self, time_step):
    if not time_step.first():
      assert time_step.discount in (0, 1), time_step.discount
    obs = time_step.observation
    obs = obs.flatten()
    obs = dict(obs) if self._obs_dict else {self._obs_key: obs}
    if 'reward' in obs:
      obs['obs_reward'] = obs.pop('reward')
    for key in self._obs_empty:
      del obs[key]
    return dict(
        reward=np.float32(0.0 if time_step.first() else time_step.reward),
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=False if time_step.first() else time_step.discount == 0,
        **obs,
    )

  def _convert(self, space):
    if hasattr(space, 'num_values'):
      return embodied.Space(space.dtype, (), 0, space.num_values)
    elif hasattr(space, 'minimum'):
      assert np.isfinite(space.minimum).all(), space.minimum
      assert np.isfinite(space.maximum).all(), space.maximum
      flat_shape = (int(np.prod(space.shape)),)
      return embodied.Space(
          space.dtype, flat_shape, space.minimum, space.maximum)
    else:
      flat_shape = (int(np.prod(space.shape)),)
      return embodied.Space(space.dtype, flat_shape, None, None)


