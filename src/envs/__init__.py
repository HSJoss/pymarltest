from functools import partial
#from envs.multiagentenv import MultiAgentEnv
from smac.env import MultiAgentEnv, StarCraft2Env
from envs.atarienv import AtariAgentEnv

import sys
import os

REGISTRY = {}

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

REGISTRY["atari"] = partial(env_fn, env=AtariAgentEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
