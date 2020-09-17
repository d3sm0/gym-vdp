import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.utils import seeding

from gym_vdp.simulator import VanDerPolPendulum
from gym_vdp.wrappers import RandomizeVDP, Constrained

_metadata = {'render.modes': ["static", "human", "rgb_array"]}

try:
    from gym.envs.classic_control import rendering
except ImportError as e:
    Warning("Live Rendering not available", e)
    _metadata['render.modes'] = ["static"]


class VanDerPolPendulumEnv(gym.Env):
    metadata = _metadata

    def __init__(self, x0=None):
        self.viewer = None
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
        if mode == "static":
            fig = self._render_static(kwargs)
            return fig
        elif mode in self.metadata["render.modes"]:
            self._render_rgb()
            return self.viewer.render(return_rgb_array=mode == "rgb_array")
        else:
            raise Exception(f"Render mode {mode} not found.")

    def _render_rgb(self):
        if self.viewer is None:
            window_size = (600, 600)
            self.viewer = rendering.Viewer(*window_size)
            self.viewer.set_bounds(-5, 5, -5, 5)

            s, _ = self._xs[0]
            circ = _render_state(s, color=(1, 0, 0))
            self.viewer.add_geom(circ)
        if len(self._xs) > 2:
            s, _ = self._xs[-2]
            line = _render_last_state(s)
            self.viewer.add_geom(line)
        s, _ = self._xs[-1]
        circ = _render_state(s, color=(0, 1, 0))
        self.viewer.add_onetime(circ)

    def _render_static(self, kwargs):
        xs, a = list(zip(*self._xs))
        xs = np.array(xs).T
        a = np.array(a)
        fig = _static_render(xs, a)
        if "fname" in kwargs.keys():
            plt.savefig(kwargs["fname"])
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        return fig

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def _render_state(s, color=(1, 0, 0)):
    attr = rendering.Transform(translation=tuple(s))
    circ = rendering.make_circle(0.1)
    circ.set_color(*color)
    circ.add_attr(attr)
    return circ


def _render_last_state(s):
    ds = 0.05
    square = ((0, 0), (ds, ds))
    attr = rendering.Transform(translation=tuple(s))
    line = rendering.make_polyline(square)
    line.set_linewidth(4)
    line.set_color(0, 0, 1)
    line.add_attr(attr)
    return line


def _static_render(xs, a):
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].set(
        xlabel="t", ylabel="y", title="time_dynamics",
    )
    ax[0].plot(xs[0], ls="--")
    ax[0].plot(xs[1], ls="-")
    ax[1].set(
        xlabel="q", ylabel="q_dot", title="state_dynamics",
    )
    ax[1].plot(xs[0], xs[1], ls="--")
    ax[1].plot(xs[0, 0], xs[1, 0], "o", c="r")
    ax[1].plot(xs[0, -1], xs[1, -1], "o", c="y")
    ax[2].set(
        xlabel="t", ylabel="u", title="controls",
    )
    ax[2].plot(a)
    plt.tight_layout()
    return fig


def make_vdp(env_id, randomize=False, constrained=False, x0=None):
    env = VanDerPolPendulumEnv(x0=x0)
    if randomize:
        env = RandomizeVDP(env)
    if constrained:
        env = Constrained(env)
    return env
