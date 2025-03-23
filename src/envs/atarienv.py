from envs.multiagentenv import MultiAgentEnv

import gymnasium as gym
import ale_py

from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import StickyAction
from gymnasium.wrappers import ClipReward
from gymnasium.wrappers import TransformReward
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import TimeLimit

import numpy as np

from utils.record import record
#from gymnasium.utils.save_video import save_video

# def make_atari_env(**kwargs):
#     gym.register_envs(ale_py)

#     # Initialise the environment
#     env = gym.make(kwargs["env_id"], render_mode="rgb_array", frameskip=1, repeat_action_probability=0.25)
#     env = AtariPreprocessing(env, grayscale_newaxis=False)
#     """env = StickyAction(env, repeat_action_probability=0.25)
#     # `ale` 인터페이스 접근
#     ale = env.unwrapped.ale
#     repeat_prob = ale.getFloat("repeat_action_probability")
#     print(f"repeat_action_probability: {repeat_prob}")"""
#     env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
#     env = TransformReward(env, lambda r: np.sign(float(r)))
#     env = FrameStackObservation(env, stack_size=kwargs["stack_size"])
#     env = TimeLimit(env, max_episode_steps=kwargs["max_episode_steps"])
#     observation, info = env.reset()
#     return env

class AtariAgentEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if self.kwargs["max_episode_steps"] == float("inf"):
            self.kwargs["max_episode_steps"] = 9999999
        self.n_agents = 1
        self.env = self._build_env()
        self.env_info = self.get_env_info()
        self.action_space = self.env.unwrapped.get_action_meanings()
        self.FIRE_action_indices = [index for index, action_name in enumerate(self.action_space) if "FIRE" in action_name]
        self.episode_limit = self.env_info["episode_limit"]

    def _build_env(self):
        gym.register_envs(ale_py)

        # Initialise the environment
        env = gym.make(self.kwargs["env_id"], render_mode="rgb_array", frameskip=1, repeat_action_probability=0.25)
        env = AtariPreprocessing(env, grayscale_newaxis=True) # obs: [H, W, C]
        """env = StickyAction(env, repeat_action_probability=0.25)
        # `ale` 인터페이스 접근
        ale = env.unwrapped.ale
        repeat_prob = ale.getFloat("repeat_action_probability")
        print(f"repeat_action_probability: {repeat_prob}")"""
        env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
        env = TransformReward(env, lambda r: np.sign(float(r)))
        env = FrameStackObservation(env, stack_size=self.kwargs["stack_size"]) # obs: [S, H, W, C]
        env = TimeLimit(env, max_episode_steps=self.kwargs["max_episode_steps"])
        self.observation, info = env.reset()
        self.observation = np.expand_dims(np.transpose(self.observation, (0, 3, 1, 2)), axis=0) # [n_agents, S, C, H, W]

        self.total_lives = info["lives"]

        return env
    
    def check_FIREFLAG(self, actions):
        if actions in self.FIRE_action_indices:
            self.FIREFLAG = True

    def init_FIREFLAG(self, current_lives):
        if self.previous_lives != current_lives:
            self.FIREFLAG = False
            self.previous_lives = current_lives
    
    def step(self, actions):
        self.observation, reward, terminated, truncated, info = self.env.step(actions)
        self.init_FIREFLAG(info["lives"])
        self.observation = np.expand_dims(np.transpose(self.observation, (0, 3, 1, 2)), axis=0) # [A, S, C, H, W]
        
        """if np.array_equal(self.previous_observation, self.observation):
            self.frozencount += 1
            if self.frozencount >= 50:
                self.FROZENFLAG = True
        else:
            self.previous_observation = self.observation
        if self.FROZENFLAG:
            print("KKK")
            truncated = True"""
        return reward, terminated, truncated, info
    
    def get_obs(self):
        return self.observation
    
    def get_obs_size(self):
        obs_shape = self.env.observation_space.shape
        return (obs_shape[0], obs_shape[3], obs_shape[1], obs_shape[2]) # [S, C, H, W]
    
    def get_avail_actions(self):
        # Init avail_actions
        #avail_actions = np.ones((self.args.n_agents, self.args.n_actions), dtype=int)
        avail_actions = np.ones((self.env_info["n_agents"], self.env_info["n_actions"]), dtype=int)

        if self.FIREFLAG:
            avail_actions[:, self.FIRE_action_indices] = 0

        return avail_actions
    
    def get_total_actions(self):
        return self.env.action_space.n
    
    def reset(self, seed=None):
        self.FIREFLAG = False
        self.previous_lives = self.total_lives
        """self.FROZENFLAG = False
        self.frozencount = 0"""
        if seed is not None:
            self.observation, info = self.env.reset(seed=seed)
        else:
            self.observation, info = self.env.reset()
        self.observation = np.expand_dims(np.transpose(self.observation, (0, 3, 1, 2)), axis=0) # [n_agents, S, C, H, W]
        #self.previous_observation = self.observation

    def close(self):
        self.env.close()

    def save_replay(self, buffer, checkpoint_path):
        env = gym.make(self.kwargs["env_id"], render_mode="rgb_array", frameskip=1, repeat_action_probability=0.25)
        
        for idx, episode_batch in enumerate(buffer):
            record(
                seed=self.kwargs["seed"],
                environment=env,
                replay_actions_dir="results/replay_actions",
                videos_dir="results/videos",
                name_prefix=f"{self.kwargs['env_id']}_{checkpoint_path}",
                #episode_index=idx+1,
                actions=episode_batch["actions"],
                scale_factor=self.kwargs["scale_factor"],
                speed=self.kwargs["speed"],
            )

    def get_env_info(self):
        env_info = {
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.env.spec.max_episode_steps if self.env.spec.max_episode_steps is not None else 0
        }
        return env_info
    