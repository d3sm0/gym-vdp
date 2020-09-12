import numpy as np

from gym_vdp.env import make_vdp
from gym_vdp.simulator import VanDerPolPendulum


def main():
    env = make_vdp("vdp-v0", randomize=False, x0=VanDerPolPendulum.best_x0)
    env.seed(0)
    env.reset()
    while True:
        a = np.zeros((1,))
        s, r, done, _ = env.step(a)
        env.render(mode="human")
        if done:
            break
    env.render(mode="static", fname="vdp.png")
    env.close()


if __name__ == "__main__":
    main()
