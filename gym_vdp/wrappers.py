import gym
import numpy as np


class RandomizeVDP(gym.Wrapper):
    def __init__(self, env):
        super(RandomizeVDP, self).__init__(env)

    def step(self, action):
        s, r, d, info = super(RandomizeVDP, self).step(action)
        delta = self.unwrapped.np_random.uniform(0, 1)
        self.env._simulator._mu += delta / self.unwrapped.max_steps
        info["alpha"] = self.env._simulator._mu
        return s, r, d, info

    def reset(self, **kwargs):
        self.env._simulator._mu = 0
        s = super(RandomizeVDP, self).reset()
        return s


class Constrained(gym.Wrapper):
    def __init__(self, env, action_bound=1, state_bound=2):
        super(Constrained, self).__init__(env)
        self.action_space = gym.spaces.Box(
            low=-action_bound, high=action_bound, shape=(1,)
        )
        self.observation_space = gym.spaces.Box(
            low=-state_bound, high=state_bound, shape=(2,)
        )

    def step(self, action: float):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        s, r, d, info = super(Constrained, self).step(action)
        s = np.clip(s, self.observation_space.low, self.observation_space.high)
        return s, r, d, info
