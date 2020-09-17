import numpy as np


def vdp(t, x, u, mu):
    x, x_dot = x
    x_ddot = mu * x_dot * (1 - x ** 2) - x + u
    return x_dot, x_ddot


def rk4(derivs, t, y0, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.

    courtesy of ~/openai/~/acrobot.py
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(thist, y0, *args, **kwargs))
        k2 = np.asarray(derivs(thist + dt2, y0 + dt2 * k1, *args, **kwargs))
        k3 = np.asarray(derivs(thist + dt2, y0 + dt2 * k2, *args, **kwargs))
        k4 = np.asarray(derivs(thist + dt, y0 + dt * k3, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


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
        out = rk4(vdp, t_span, self.x, u, self._mu)
        x = np.clip(out[-1], -self.state_bound, self.state_bound)
        c = self.cost_fn(x, u)
        self.time_step += 1
        self.x = x
        return x, c

    @property
    def best_x0(self):
        return self._best_x0
