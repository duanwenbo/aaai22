"""
# time: 07/09/2021
# update: /
# author: Bobby
The agent in a single reinforcement learning environment
"""
import sys

from torch.distributions.utils import probs_to_logits
sys.path.append("/home/gamma/wb_alchemy/sub_project/")
import torch
from torch.distributions import Categorical
from Algorithms.PPO import PPO
from Algorithms.Differentiable_SGD import DifferentiableSGD
import matplotlib.pyplot as plt
import yaml
from matplotlib.pyplot import figure
class RLAgent:
    """for reinforcement learning"""
    def __init__(self, critic_net, actor_net, env, device, LEARNING_RATE, algo=PPO) -> None:
        self.critic_net = critic_net.to(device)
        self.actor_net = actor_net.to(device)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=LEARNING_RATE)
        self.actor_net_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=LEARNING_RATE)
        self.algo = algo
        self.device = device
        self.env = env

    def _choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        dist = Categorical(self.actor_net(state))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def sample_trajectory(self):
        states = []
        rewards = []
        actions = []
        next_states = []
        log_probs = []
        state = self.env.reset()  
        done = False
        while not done:
            action, log_prob = self._choose_action(state)  
            next_state, reward, done, distance = self.env.step(action)
            # start recording
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            log_probs.append(log_prob)
            state = next_state
        return {"states":states, "rewards": rewards, "actions": actions, 
                "next_states": next_states, "log_probs": log_probs}, sum(rewards), distance
    
    def learn(self, trajectory, EPOCH=3):
        # note this learning function (which has an inner loop) is specifc designed for PPO
        algorithm = self.algo(trajectory, self.actor_net, self.critic_net, self.device)
        old_probs = trajectory["log_probs"]
        if algorithm.name != "PPO":
            EPOCH = 1
        for _ in range(EPOCH):
            actor_loss = algorithm.actor_loss(old_probs)
            critic_loss = algorithm.critic_loss()

            self.actor_net_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_net_optimizer.step()

            self.critic_net_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_net_optimizer.step()
    
    def visualize(self, trajectory, name):
        # load environment config
        with open("/home/gamma/wb_alchemy/sub_project/Configuration/environment.yml","r") as f:
            config = yaml.load(f.read())

        # draw environment   
        plt.axis([0,10,0,10])
        obstacles = config["{}".format(self.env.obstacles)]
        x,y=[],[]
        for obstacle in obstacles:
            x.extend([i[0] for i in obstacle])
            y.extend(i[1] for i in obstacle)
        for i in range(0,len(x),2):
            plt.plot(x[i:i+2], y[i:i+2],'k')
        
        # draw goal
        goal = self.env.goal
        plt.scatter(goal[0], goal[1], "*", "r")

        # plot trajectory
        states = trajectory["states"]
        x = [state[0] for state in states]
        y = [state[1] for state in states]
        plt.scatter(x,y,"y")
        plt.savefig("{}.png".format(name))
        

    


class MetaAgent:
    """for meta reinforcement learning, one-step optimization in policy optimization"""
    def __init__(self, critic_net, actor_net, env, device, LEARNING_RATE, algo=PPO) -> None:
        self.critic_net = critic_net.to(device)
        self.actor_net = actor_net.to(device)
        self.critic_net_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=LEARNING_RATE)
        self.actor_net_optimizer = DifferentiableSGD(module=self.actor_net)
        self.algo = algo
        self.device = device
        self.env = env

    def _choose_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        dist = Categorical(self.actor_net(state))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def sample_trajectory(self):
        states = []
        rewards = []
        actions = []
        next_states = []
        log_probs = []
        state = self.env.reset()  
        done = False
        while not done:
            action, log_prob = self._choose_action(state)  
            next_state, reward, done, distance = self.env.step(action)
            # start recording
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            log_probs.append(log_prob)
            state = next_state
        return {"states":states, "rewards": rewards, "actions": actions, 
                "next_states": next_states, "log_probs": log_probs}, sum(rewards), distance
    
    def learn(self, trajectory, EPOCH=3):
        # note this learning function (which has an inner loop) is specifc designed for PPO
        algorithm = self.algo(trajectory, self.actor_net, self.critic_net, self.device)
        old_probs = trajectory["log_probs"]
        if algorithm.name != "PPO":
            EPOCH = 1
        for _ in range(EPOCH):
            actor_loss = algorithm.actor_loss(old_probs)
            self.actor_net_optimizer.set_grads_none()
            actor_loss.backward(retain_graph=True)
            with torch.set_grad_enabled(True):
                self.actor_net_optimizer.step()

            self.critic_net_optimizer.zero_grad()
            critic_loss = algorithm.critic_loss()
            critic_loss.backward()
            self.critic_net_optimizer.step()
    
    def policy_loss(self, trajectory):
        """
        one more iteration when calculating the loss for the new trajectory
        (importance sampling ?)
        """
        algorithm = self.algo(trajectory, self.actor_net, self.critic_net, self.device)
        old_probs = trajectory["log_probs"]
        actor_loss = algorithm.actor_loss(old_probs)
        return actor_loss


class TestAgent:
    def __init__(self, critic_net, actor_net, env, device) -> None:
            self.critic_net = critic_net.to(device)
            self.actor_net = actor_net.to(device)
            self.device = device
            self.env = env 
    
    def _choose_action(self,state):
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        dist = Categorical(self.actor_net(state))
        action = dist.sample()
        return action.item()
    
    def play(self):
        state = self.env.reset()  
        done = False
        states = []
        rewards = []
        actions = []
        next_states = []
        while not done:
            action = self._choose_action(state)  
            next_state, reward, done, distance = self.env.step(action)
            # start recording
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            state = next_state
        return {"states":states, "rewards": rewards, "actions": actions, 
                "next_states": next_states}, sum(rewards), distance
    
    def visualize(self, trajectory, name, marker, color):
        # load environment config
        with open("/home/gamma/wb_alchemy/sub_project/Configuration/environment.yml","r") as f:
            config = yaml.load(f.read())

        # draw environment   
        plt.axis([0,10,0,10])
        obstacles = config["{}".format(self.env.obstacles)]
        x,y=[],[]
        for obstacle in obstacles:
            x.extend([i[0] for i in obstacle])
            y.extend(i[1] for i in obstacle)
        for i in range(0,len(x),2):
            plt.plot(x[i:i+2], y[i:i+2],'k')
        
        # draw goal
        goal = self.env.goal
        plt.scatter(goal[0], goal[1], c=['#FF0000'], marker="*", s=300)

        # plot trajectory
        states = trajectory["states"]
        x = [state[0] for state in states]
        y = [state[1] for state in states]
        plt.scatter(x,y,c=color, s=5, marker=marker)
        


if __name__ == "__main__":
    with open("/home/gamma/wb_alchemy/sub_project/Configuration/environment.yml","r") as f:
        config = yaml.load(f.read())   
    obstacles = config["None"]
    x,y=[],[]
    for obstacle in obstacles:
        x.extend([i[0] for i in obstacle])
        y.extend(i[1] for i in obstacle)
    plt.axis([0,10,0,10])
    for i in range(0,len(x),2):
        plt.plot(x[i:i+2], y[i:i+2],'k')
    plt.savefig("environment_null.png")