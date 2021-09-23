"""
# time: 09/09/2021
# update: 20/09/2021
# author: Bobby
an implementation of MAML Algorithm,
currently inner loop optimization is based on VPG.
"""
from Entities.agent import MetaAgent
import torch
from Networks.Discrete import Actor_net, Critic_net
from Entities.environment_sampler import Environment_sampler
from Entities.clone_module import clone_module
from copy import deepcopy
import csv
from Algorithms.VPG import VPG

EPISODE_LENGTH = 20000
INNER_LEARNING_RATE = 0.0003
OUTER_LEARNING_RATE = 0.0005
STEP_NUM = 1


def train():
    try:
        #### initialization####
        # initialize test environments
        environments = Environment_sampler(
            see_goal=True, obstacles="Medium").multi_env(
            env_num=8, goal_distribution="average")
        # initialize policy net and baseline net
        action_space, observation_space = environments[
            0].action_space.n, environments[0].observation_space.shape[0]
        policy_net = Actor_net(input=observation_space,
                               hidden=128,
                               output=action_space)
        baseline_net = Critic_net(input=observation_space,
                                  hidden=128,
                                  output=1)
        policy_optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=OUTER_LEARNING_RATE)  # essential parameters

        #### main training loop####
        for i in range(EPISODE_LENGTH):
            cumulative_loss = 0.  # for recording policy losses of each tasks in one episode
            cumulative_distance = 0.  # for evaluating arverage effect in one episode, only used in this environment
            # cumulative_rewards = 0.  #  for evaluating the mete_learner

            for j, environment in enumerate(environments):
                meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                        actor_net=clone_module(policy_net), # use clone_module() to keep the computational graph !
                                        env=environment,
                                        device="cpu",
                                        LEARNING_RATE=INNER_LEARNING_RATE,
                                        algo=VPG)

                # STEP_NUM is set as 1 by default
                # increaseing STEP_NUM could increase the training stability
                # but need more time
                for _ in range(STEP_NUM):
                    trajectory, ep_reward, distance = meta_leaner.sample_trajectory()
                    meta_leaner.learn(trajectory)
                new_trajectory, ep_reward, distance = meta_leaner.sample_trajectory()
                one_step_opt_loss = meta_leaner.policy_loss(new_trajectory)
                cumulative_loss += one_step_opt_loss
                cumulative_distance += distance

                print("task:{}, distance:{}".format(j, round(distance, 2)))

            policy_optimizer.zero_grad()
            cumulative_loss.backward()
            policy_optimizer.step()

            ep_distance = round(cumulative_distance / len(environments), 2)
            print("##############################")
            print("episode:{}  distance:{}".format(i, ep_distance))
            print("##############################")

            with open("/home/gamma/wb_alchemy/sub_project/Chongkai/0916_night_02.csv", "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    ["maml_wide_distribution", "VPG", i, ep_distance])
        torch.save(policy_net, "maml_02.pkl")
    except BaseException:
        torch.save(policy_net, "maml_02.pkl")
        print("done")


if __name__ == "__main__":
    train()
