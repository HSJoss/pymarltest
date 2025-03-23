REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .atari_episode_runner import AtariEpisodeRunner
REGISTRY["atari_episode"] = AtariEpisodeRunner

from .atari_parallel_runner import AtariParallelRunner
REGISTRY["atari_parallel"] = AtariParallelRunner

from .origin_atari_episode_runner import OriginAtariEpisodeRunner
REGISTRY["origin_atari_episode"] = OriginAtariEpisodeRunner

from .origin_atari_parallel_runner import OriginAtariParallelRunner
REGISTRY["origin_atari_parallel"] = OriginAtariParallelRunner
