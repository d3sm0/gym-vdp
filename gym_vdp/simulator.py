import numpy as np
try:
    from vdp import rk4
except ImportError:
    # Using numpy rk4 integration. For improvement check rust version."
    from gym_vdp.utils import rk4


class VanDerPolPendulum:
    _best_x0 = [-0.1144, 2.0578]  # taken from drake, thanks !

    def __init__(self, x0=(0, 1), mu=1, state_bound=10., action_bound=10.):
        self._dt = 0.1
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
        out = rk4(t_span, self.x, u, self._mu)
        x = np.clip(out, -self.state_bound, self.state_bound)
        c = self.cost_fn(x, u)
        self.time_step += 1
        self.x = x
        return x, c

    @property
    def best_x0(self):
        return self._best_x0
