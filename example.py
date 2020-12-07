import numpy as np

from gym_vdp.env import make_vdp


def control():
    env = make_vdp("vdp-v0", randomize=True, x0=(1, 1))
    # env = Monitor(env, directory="logs/", force=True)
    env.seed(0)
    env.reset()
    while True:
        a = np.random.uniform(-2, 2)
        s, r, done, info = env.step(a)
        env.render(mode="human")
        if done:
            break
    env.render(mode="static", fname="figures/vdp.png")
    env.close()


def stationarity():
    env = make_vdp("vdp-lc-v0", randomize=False)
    # env = Monitor(env, directory="logs/", force=True)
    env.seed(0)
    env.reset()
    while True:
        a = env.unwrapped._simulator._best_action
        s, r, done, info = env.step(a)
        # env.render(mode="human")
        if done:
            break
    env.render(mode="static", fname="figures/vdp.png")
    env.close()


if __name__ == "__main__":
    stationarity()
    # control()
