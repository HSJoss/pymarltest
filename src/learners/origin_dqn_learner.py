import copy
from components.episode_buffer import EpisodeBatch
#from components.s_episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
from torch.nn import functional as F


class OriginDQNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions"][:]
        rewards = batch["reward"][:]
        terminated = batch["terminated"][:].float()
        """print("rewards", rewards.shape)
        print("actions", actions.shape)
        print("terminated", terminated.shape)
        print("mask", mask.shape)"""

        # Calculate estimated Q-Values
        mac_out = []
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        #print("m",mac_out.shape)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:], dim=3, index=actions).squeeze(3)  # Remove the last dim
        #print("q",chosen_action_qvals.shape)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward2(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Mask out unavailable actions
        #target_mac_out[avail_actions[:] == 0] = -9999999
        
        target_max_qvals = target_mac_out.max(dim=3)[0]
        
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        #chosen_action_qvals = chosen_action_qvals * mask
        #targets = targets * mask

        # Normal L2 loss, take mean over actual data
        #loss = F.smooth_l1_loss(chosen_action_qvals, targets)
        #loss = F.smooth_l1_loss(chosen_action_qvals, targets.detach())
        loss = (td_error ** 2).sum() / td_error.size(0)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            #self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals).sum().item()/(self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets).sum().item()/(self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        #if self.mixer is not None:
            #self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        #if self.mixer is not None:
            #self.mixer.cuda()
            #self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        #if self.mixer is not None:
            #th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        #if self.mixer is not None:
            #self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
