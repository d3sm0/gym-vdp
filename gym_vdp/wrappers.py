import gym
import numpy as np


class RandomizeVDP(gym.Wrapper):
    def __init__(self, env, std=0.03):
        super(RandomizeVDP, self).__init__(env)
        self._std = std

    def step(self, action):
        s, r, d, info = super(RandomizeVDP, self).step(action)
        delta = self.unwrapped.np_random.normal(scale=self._std)
        self.env._simulator._mu += abs(delta)
        info["param"] = self.env._simulator._mu
        return s, r, d, info

    def reset(self, **kwargs):
        self.env._simulator._mu = self.unwrapped.np_random.normal(1.,scale=self._std)
        s = super(RandomizeVDP, self).reset()
        return s


class Constrained(gym.Wrapper):
    def __init__(self, env, action_bound=5., state_bound=10.):
        super(Constrained, self).__init__(env)
        if action_bound is not None:
            self.action_space = gym.spaces.Box(
                low=-action_bound, high=action_bound, shape=(1,)
            )
        if state_bound is not None:
            self.observation_space = gym.spaces.Box(
                low=-state_bound, high=state_bound, shape=(2,)
            )

    def step(self, action: float):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        s, r, d, info = super(Constrained, self).step(action)
        s = np.clip(s, self.observation_space.low, self.observation_space.high)
        return s, r, d, info
