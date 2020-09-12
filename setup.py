from setuptools import setup

setup(
    name="gym_vdp",
    version="0.1",
    keywords="van der pool, control, environment, agent, rl, openaigym, openai-gym, gym",
    url="https://github.com/d3sm0/gym-vdp",
    description="benchmark control task for rl",
    packages=["gym_vdp"],
    install_requires=
    [
        "gym>=0.17.2",
        "numpy>=1.19.1",
        "matplotlib>=3.3.0"
    ]
)
