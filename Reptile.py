"""
# time: 10/09/2021
# update: /
# author: Bobby
an implementation of Reptile Algorithm
"""

from Entities.agent import MetaAgent
import torch
from Networks.Discrete import Actor_net, Critic_net
from Entities.environment_sampler import Environment_sampler
from Entities.clone_module import clone_module
import yaml
from copy import deepcopy
import wandb
import csv
from Algorithms.PPO import PPO
from Algorithms.VPG import VPG
from copy import deepcopy
import time
from Algorithms.Differentiable_SGD import DifferentiableSGD
from torch import autograd
import torch.nn as nn
from copy import deepcopy


EPISODE_LENGTH = 3000
INNER_LEARNING_RATE = 0.0003
OUTER_LEARNING_RATE = 0.0005
K = 3


def train(n):
    # initialize test environments
    environments = Environment_sampler(see_goal=True, 
                                    obstacles="Medium").multi_env(env_num=8,
                                                                    goal_distribution="average")
    # discrete action space env 
    action_space, observation_space = environments[0].action_space.n, environments[0].observation_space.shape[0]
    # create policy net and baseline net
    policy_net = Actor_net(input=observation_space,
                        hidden=128,
                        output=action_space)
    baseline_net = Critic_net(input=observation_space,
                            hidden=128,
                            output=1)
    # policy_optimizer = torch.optim.Adam(policy_net.parameters(), 
    #                                     lr=OUTER_LEARNING_RATE)# essential parameters
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), 
                                            lr=OUTER_LEARNING_RATE)# essential parameters
    # policy_optimizer = DifferentiableSGD(module=policy_net, lr=0.001)


    # main training loop
    for i in range(EPISODE_LENGTH):
        cumulative_distance = 0.

        weight_before = deepcopy(policy_net.state_dict())
        # initialize a dict
        weight_difference = weight_before.fromkeys(weight_before,0)
        for j, environment in enumerate(environments):
            policy_net.load_state_dict(weight_before)
            meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                actor_net=policy_net,
                                env=environment,
                                device="cpu",
                                LEARNING_RATE=INNER_LEARNING_RATE,
                                algo=VPG)
            # expeiment 3: Reptile
            # trajectory,ep_reward,distance = meta_leaner.sample_trajectory()
            # loss = meta_leaner.policy_loss(trajectory)

            sub_meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                actor_net=deepcopy(policy_net),
                                env=environment,
                                device="cpu",
                                LEARNING_RATE=INNER_LEARNING_RATE,
                                algo=VPG)

            policy_optimizer.zero_grad()
            for _ in range(K):
                trajectory,ep_reward,distance = sub_meta_leaner.sample_trajectory()
                current_loss = meta_leaner.policy_loss(trajectory)
                current_loss.backward()
                policy_optimizer.step()
                # for param in policy_net.parameters():
                #     param.data -= INNER_LEARNING_RATE * param.grad.data
            # loss1 = meta_leaner.policy_loss(trajectory)
            # policy_optimizer.zero_grad()
            # loss1.backward()
            # policy_optimizer.step()

            # loss2 = meta_leaner.policy_loss(trajectory)
            # policy_optimizer.zero_grad()
            # loss2.backward()
            # policy_optimizer.step()

            # loss3 = meta_leaner.policy_loss(trajectory)
            # policy_optimizer.zero_grad()
            # loss3.backward()
            # policy_optimizer.step()

            weight_after = policy_net.state_dict()
            weight_difference = {name: weight_difference[name] + weight_after[name] - weight_before[name] for name in weight_difference}
            cumulative_distance += distance

            print("task:{}, distance:{}".format(j, round(distance,2)))
        
        policy_net.load_state_dict({name: weight_before[name] + weight_difference[name] * (INNER_LEARNING_RATE/8) for name in weight_difference})

        ep_distance = round(cumulative_distance/len(environments),2)
        print("##############################")
        print("episode:{}  distance:{}".format(i, ep_distance))
        print("##############################")

        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/reptile.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Reptile", "VPG",  i, ep_distance, n])

    torch.save("reptile.pkl")



if __name__ == "__main__":
    for i in range(3):
        t1 = time.time()
        train(i)
        t2 = time.time()
        td = t2-t1
        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/reptile.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Reptile_time", td])
            
   
