import functools
import os

import numpy as np
from scipy.integrate import odeint

# try:
#    from vdp import rk4
# except ImportError:
# Using numpy rk4 integration. For improvement check rust version."
from gym_vdp.utils import rk4


class VanDerPolOscillator:
    _best_x0 = [-0.1144, 2.0578]  # taken from drake, thanks !

    def __init__(self, x0=(0, 1), mu=1, state_bound=10., action_bound=2., dt=0.01):
        self._dt = dt
        self.time_step = 0
        self.state_bound = state_bound
        self.action_bound = action_bound

        self._mu = mu
        self.x = self._x0 = x0

    def reset(self, x0=None):
        self.time_step = 0
        self.x = self._x0 if x0 is None else x0
        return self.x

    def cost_fn(self, x, u):
        return np.linalg.norm(u) ** 2 + np.linalg.norm(x) ** 2

    def step(self, u):
        u = float(np.clip(u, -self.action_bound, self.action_bound))
        t_span = [0, self._dt]
        out = rk4(_vdp, t_span, self.x.copy(), u, self._mu)
        x = np.clip(out, -self.state_bound, self.state_bound)
        c = self.cost_fn(self.x, u)
        self.time_step += 1
        self.x = x
        return x, c

    @property
    def best_x0(self):
        return self._best_x0


class LimitCycleVDP(VanDerPolOscillator):
    # Taken from Practical Bifurcation and Stability Analysis, page 347 Seydel R. Forced Van der Pol oscillator
    _x0 = [1.53457323, -0.18991305]
    _period = 10.4719755

    def __init__(self, gamma=0.5, mu=4, u=1.8, max_steps=500):
        path = os.path.join(os.path.dirname(__file__), "solution.npy")
        if not os.path.exists(path):
            times = np.linspace(0, self._period, max_steps)
            space, dt = _simulate_optimal_trajectory(self._x0, times, u, mu, gamma)
            np.save(path, (space, dt))
        else:
            space, dt = np.load(path, allow_pickle=True)
        assert len(space) == max_steps, "saved trajectory does not match max_steps"
        self.space = space
        self.gamma = gamma
        self.max_steps = len(self.space) - 1

        self._best_action = u
        super().__init__(self._x0, mu=mu, dt=dt, action_bound=2)

    def cost_fn(self, x, u):
        return np.linalg.norm(x - self.space[self.time_step]) ** 2 / self.max_steps + np.linalg.norm(
            u - self._best_action) ** 2

    def reset(self, x0=None):
        return self._x0


def _vdp(t, x, u, mu, gamma=0.5):
    x, x_dot = x
    x_ddot = mu * x_dot * (1 - x ** 2) - x + gamma * np.cos(u * t)
    return x_dot, x_ddot


def _vdp_ode(x, t, u, mu, gamma):
    return _vdp(t, x, u, mu, gamma)


def _simulate_optimal_trajectory(x0, times, u, mu, gamma):
    dt = times[1] - times[0]
    fn = functools.partial(_vdp_ode, u=u, mu=mu, gamma=gamma)
    space = odeint(fn, x0, t=times)
    return space, dt
