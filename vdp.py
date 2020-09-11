import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from gym.utils import seeding


class VanDerPolPendulum:
    _best_x0 = [-0.1144, 2.0578]  # taken from drake, thanks !

    def __init__(self, x0=(0, 1), mu=1):
        self._dt = 0.1
        self.time_step = 0

        self._mu = mu
        self.x = self._x0 = x0

    def reset(self, x0=None):
        self.time_step = 0
        self.x = self._x0 if x0 is None else x0
        return self.x

    def cost_fn(self, x, u):
        return np.linalg.norm(u) ** 2 + np.linalg.norm(x) ** 2

    def step(self, u):
        t_span = [0, self._dt]
        out = scipy.integrate.solve_ivp(vdp, t_span, self.x, args=(u, self._mu))
        x = out.y[:, -1]
        c = self.cost_fn(x, u)
        self.time_step += 1
        self.x = x
        return x, c


def vdp(t, x, u, mu):
    x, x_dot = x
    x_ddot = mu * x_dot * (1 - x ** 2) - x + u
    return x_dot, x_ddot


class VanDerPolPendulumEnv(gym.Env):
    def __init__(self, x0=None):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-3, high=3, shape=(2,))
        self._simulator = VanDerPolPendulum(x0)
        self._xs = []
        self.max_steps = 200
        self.seed()

    def step(self, action: float):
        s, c = self._simulator.step(action)
        done = self.max_steps == self._simulator.time_step
        self._xs.append((s, action))
        return s, -c, done, {}

    def reset(self):
        self._xs = []
        x0 = self.observation_space.sample()
        s = self._simulator.reset(x0)
        self._xs.append((s, np.zeros(shape=(1,))))
        return s

    def render(self, mode="human", **kwargs):
        xs, a = list(zip(*self._xs))
        xs = np.array(xs).T
        a = np.array(a)
        fig, ax = plt.subplots(1, 3, figsize=(10,4))
        ax[0].set(
            xlabel="t",
            ylabel="y",
            title="time_dynamics",
        )
        ax[0].plot(xs[0], ls='--')
        ax[0].plot(xs[1], ls='-')

        ax[1].set(
            xlabel="q",
            ylabel="q_dot",
            title="state_dynamics",
        )
        ax[1].plot(xs[0], xs[1], ls='--')
        ax[1].plot(xs[0, 0], xs[1, 0], 'o', c='r')
        ax[1].plot(xs[0, -1], xs[1, -1], 'o', c='y')

        ax[2].set(
            xlabel="t",
            ylabel="u",
            title="controls",
        )
        ax[2].plot(a)
        plt.tight_layout()
        return fig

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]


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


def make_vdp(env_id, randomize=True, x0=None):
    env = VanDerPolPendulumEnv()
    if randomize:
        env = RandomizeVDP(env)
    return env


def _test_env():
    env = make_vdp("vdp-v0", randomize=False, x0=VanDerPolPendulum._best_x0)
    env.seed(0)
    env.reset()
    while True:
        a = np.zeros((1,))
        s, r, done, _ = env.step(a)
        if done: break
    env.render()
    plt.savefig("test_vdp")


if __name__ == '__main__':
    _test_env()
