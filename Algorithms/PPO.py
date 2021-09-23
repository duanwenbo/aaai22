"""
# time: 07/09/2021
# update: /
# author: Bobby
A sinple version of PPO with generaliz advantage estimater.
"""

import torch
from torch.distributions import Categorical
import numpy as np

class PPO():
    def __init__(self, trajectory, policy_net, critic_net, device) -> None:
        self.states = trajectory["states"]
        self.rewards = trajectory["rewards"]
        self.actions = trajectory["actions"]
        self.device = device
        self.policy_net = policy_net # the device of model has been assigned in outer file
        self.critic_net = critic_net
        self.name = "PPO"
    
    def _reward2go(self, GAMMA=0.98) -> torch.Tensor:
        """
        reward to go: [R0+gamma*R1+gamma^2*R2, R1+gamma*R2, R2]
        """
        rtg = np.zeros_like(self.rewards) 
        for i in reversed(range(len(self.rewards))):
            rtg[i] = self.rewards[i] + ( GAMMA*rtg[i+1] if i+1 < len(self.rewards) else 0 ) 
        rtg = torch.tensor(rtg, dtype=torch.float32).to(self.device)
        return rtg
    
    def _advantage_function(self, GAMMA=0.98, GAE_LAMBDA=0.97) -> torch.Tensor:
        """
        Generalize Advantage Estimator
        """
        # prepare data, convert data to numpy
        states_tensor = torch.tensor(self.states,dtype=torch.float32).to(self.device)
        state_values = self.critic_net(states_tensor).squeeze().detach().cpu().numpy()
        rewards = np.array(self.rewards)

        # compute delta, note the result is one unit shorter since to form the TD difference
        delta = rewards[:-1] + GAMMA * state_values[1:] - state_values[:-1]

        # compute discounted accumulation
        for i in reversed(range(len(delta))):
            delta[i] = delta[i] + GAMMA * GAE_LAMBDA * delta[i + 1] if i + 1 < len(delta) else delta[i]
        gae_advantage = torch.as_tensor(delta, dtype=torch.float32).to(self.device)
        return gae_advantage 
    
    def _prob_ratio(self, old_probs) -> torch.Tensor:
        """
        old_prob: list, log probabilities before model optimization
        """
        # shorten one unit to match the lenghth of advantage
        states = torch.tensor(self.states[:-1], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions[:-1], dtype=torch.float32).to(self.device)
        new_probs = Categorical(self.policy_net(states)).log_prob(actions)
        old_probs = torch.tensor(old_probs[:-1], dtype=torch.float32).to(self.device)
        prob_ratio = new_probs.exp() / old_probs.exp()
        return prob_ratio

    def critic_loss(self) -> torch.Tensor:
        """
        critic loss: MSE(reward2go, estimate state value)
        """
        rtg = self._reward2go()
        states_tensor = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        state_values = self.critic_net(states_tensor).squeeze()
        critic_loss = (rtg - state_values) ** 2
        critic_loss = critic_loss.mean()
        return critic_loss
    
    def actor_loss(self, old_probs, EPSILON=0.2) -> torch.Tensor:
        prob_ratio = self._prob_ratio(old_probs)
        advantage = self._advantage_function()
        assert len(prob_ratio) == len(advantage), "check length please"
        clip_loss = - torch.min(prob_ratio * advantage, torch.clamp(prob_ratio, 1 - EPSILON, 1 + EPSILON) * advantage).mean()
        return clip_loss