REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .ddpg_parallel_runner import DDPG_ParallelRunner
REGISTRY["ddpg_parallel"] = DDPG_ParallelRunner
