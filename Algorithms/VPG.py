"""
# time: 09/09/2021
# update: /
# author: Bobby
A sinple version of VPG with generaliz advantage estimater.
"""

import torch
import numpy as np

class VPG():
    def __init__(self, trajectory, policy_net, critic_net, device) -> None:
        self.states = trajectory["states"]
        self.rewards = trajectory["rewards"]
        self.actions = trajectory["actions"]
        self.log_probs = trajectory["log_probs"]
        self.device = device
        self.policy_net = policy_net # the device of model has been assigned in outer file
        self.critic_net = critic_net
        self.name = "VPG"
    
    def _reward2go(self, GAMMA=0.98) -> torch.Tensor:
        """
        reward to go: [R0+gamma*R1+gamma^2*R2, R1+gamma*R2, R2]
        """
        rtg = np.zeros_like(self.rewards) 
        for i in reversed(range(len(self.rewards))):
            rtg[i] = self.rewards[i] + ( GAMMA*rtg[i+1] if i+1 < len(self.rewards) else 0 ) 
        rtg = torch.tensor(rtg, dtype=torch.float32).to(self.device)
        return rtg
    
    def _advantage_function(self, GAMMA=0.98, GAE_LAMBDA=0.97) -> list:
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
        return delta 

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
    
    def actor_loss(self,*args) -> torch.Tensor:
        policy_loss = 0.
        advantages = self._advantage_function()
        for log_prob, advantage in zip(self.log_probs[:-1], advantages):
            policy_loss = policy_loss + log_prob * advantage
        policy_loss =  - policy_loss.mean()
        assert type(policy_loss) == torch.Tensor, "data type error, suppose getting a tensor"
        return policy_loss