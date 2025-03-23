from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
#from components.s_episode_buffer import EpisodeBatch
import numpy as np
import pickle


class OriginAtariEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self, buffer):
        self.env.save_replay(buffer, self.args.checkpoint_path)

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        if self.args.save_replay == True:
            self.env.reset(seed=self.args.seed)
        else:
            self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        truncated = False
        episode_return = 0
        episode_batch = []

        while not terminated and not truncated:
            self.batch = self.new_batch()

            pre_transition_data = {
                "obs": [self.env.get_obs()],
                "avail_actions": [self.env.get_avail_actions()],
            }

            self.batch.update(pre_transition_data, ts=0)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=0, t_env=self.t_env, test_mode=test_mode)
            ##actionss.append(actions[0])
            self.env.check_FIREFLAG(actions[0])

            reward, terminated, truncated, info = self.env.step(actions[0])

            episode_return += reward
            #print(info)
            
            #print(self.t)
            #print("terminated",terminated)
            #print("truncated",truncated)

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated or truncated,)],
                "next_obs": [self.env.get_obs()],
            }

            self.batch.update(post_transition_data, ts=0)

            self.t += 1

            episode_batch.append(self.batch)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats["death"] = (self.env.total_lives - info["lives"]) + cur_stats.get("death", 0)
        cur_stats["n_frame_number"] = info["episode_frame_number"] + cur_stats.get("n_frame_number", 0)
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        
        return episode_batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
