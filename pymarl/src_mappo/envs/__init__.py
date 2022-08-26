import os
import sys

# refer upper directory (./smac_plus)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


from functools import partial
from smac_plus.env import MultiAgentEnv, StarCraft2Env_Plus, StarCraft2Env

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

for arg in sys.argv:
    if 'smac_plus' in arg:
        REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env_Plus)
        break
    else:
        REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
        


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))