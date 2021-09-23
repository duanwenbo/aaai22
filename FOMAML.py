"""
# time: 09/09/2021
# update: /
# author: Bobby
an implementation of FOMAML Algorithm
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
from Algorithms.Differentiable_SGD import DifferentiableSGD
import time

EPISODE_LENGTH = 20000
INNER_LEARNING_RATE = 0.0003
OUTER_LEARNING_RATE = 0.0005

def train():
    try:
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
        policy_optimizer = torch.optim.Adam(policy_net.parameters(), 
                                            lr=OUTER_LEARNING_RATE)# essential parameters
        # policy_optimizer = DifferentiableSGD(module=policy_net)


        # main training loop
        for i in range(EPISODE_LENGTH):
            cumulative_distance = 0.
            # cumulative_rewards = 0.  # used for evaluating the mete_learner
            for j, environment in enumerate(environments):
                meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                    actor_net=policy_net,
                                    env=environment,
                                    device="cpu",
                                    LEARNING_RATE=INNER_LEARNING_RATE,
                                    algo=VPG)

                # expeiment 2: FOMAML
                trajectory,ep_reward,distance = meta_leaner.sample_trajectory()
                loss = meta_leaner.policy_loss(trajectory)

                # policy_optimizer.set_grads_none()
                # loss.backward()
                # with torch.set_grad_enabled(True):
                #     policy_optimizer.step()
                policy_net.zero_grad()
                loss.backward()
                policy_optimizer.step()

                cumulative_distance += distance

                print("task:{}, distance:{}".format(j, round(distance,2)))
            
            # policy_optimizer.set_grads_none()
            # loss.backward()
            # with torch.set_grad_enabled(True):
            #     policy_optimizer.step()

            trajectory,ep_reward,distance = meta_leaner.sample_trajectory()  ## SAVE THE BEFORE WEIGHT
            loss = meta_leaner.policy_loss(trajectory)


            policy_net.zero_grad()
            loss.backward()
            policy_optimizer.step()

            ep_distance = round(cumulative_distance/len(environments),2)
            print("##############################")
            print("episode:{}  distance:{}".format(i, ep_distance))
            print("##############################")

            with open("/home/gamma/wb_alchemy/sub_project/Chongkai/fomaml_0916.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(["FOMAML", "VPG",  i, ep_distance])

        torch.save(policy_net, "fomaml.pkl")
    except BaseException:
        torch.save(policy_net, "fomaml.pkl")
        print("done")
    
for _ in range(1):
    t1 = time.time()
    train()
    t2 = time.time()
    td = t2 -t1
    with open("/home/gamma/wb_alchemy/sub_project/Chongkai/fomaml_0916.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["FOMAML_time", td])
