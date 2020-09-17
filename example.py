import numpy as np

from gym_vdp.env import make_vdp
from gym_vdp.simulator import VanDerPolPendulum
from gym.wrappers.monitor import Monitor


def main():
    env = make_vdp("vdp-v0", randomize=True, x0=VanDerPolPendulum.best_x0)
    env = Monitor(env, directory="logs/", force=True)
    env.seed(0)
    env.reset()
    while True:
        a = np.random.uniform(0,1)
        s, r, done, info = env.step(a)
        print(info)
        #env.render(mode="human")
        if done:
            break
    #env.render(mode="static", fname="figures/vdp.png")
    env.close()


if __name__ == "__main__":
    main()
