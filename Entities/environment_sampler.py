"""
# time: 09/09/2021
# update: /
# author: Bobby
Creating tasks for both RL and Meta_RL according to different requirements
"""
import sys
sys.path.append("/home/gamma/wb_alchemy/sub_project/")
import random
from _Environment.navigationPro import NavigationPro
from _Environment.navigation import Navigation_v0

import gym

class Environment_sampler:
    def __init__(self, see_goal, obstacles) -> None:
        self.see_goal = see_goal
        self.obstacles = obstacles
        self.env = NavigationPro
        random.seed(234)  # 128

    def single_env(self, goal):
        """for a single env test in reinforcement learning"""
        env = self.env(goal, self.see_goal, self.obstacles)
        # env = self.env(goal=goal, see_goal=self.see_goal, obstacles=self.obstacles)
        return env
    
    def single_env_gym(env="CartPole-v1"):
        """for validation purpose"""
        return gym.make(env)
        
    def multi_env(self, env_num=5, goal_distribution="average"):
        """for MAML training"""
        goal_list = []
        if goal_distribution == "average":
            """SPECIFICALLY designed for 'Medium' hard level env"""
            """make sure the goal fall evenly in each quadrant"""
            goal_generator = lambda: [
            (round(random.uniform(5,10),1), round(random.uniform(5,10),1)), # quadrant1
            (round(random.uniform(0,5),1), round(random.uniform(5,10),1)),  # quadrant2
            (round(random.uniform(0,5),1), round(random.uniform(0,5),1)),   # quadrant3
            (round(random.uniform(5,10),1), round(random.uniform(0,5),1))]  # quadrant4
            for _ in range(env_num//4):
                goal_list.extend(goal_generator())
            for i in range(env_num%4):
                goal_list.append(goal_generator()[i])
            env_list = [self.single_env(goal) for goal in goal_list]
            return env_list
        
        elif goal_distribution == "random":
            goal = lambda: (round(random.uniform(0,10),1), round(random.uniform(0,10),1))
            env_list = [self.single_env(goal()) for _ in range(env_num)]
            return env_list
        
        elif goal_distribution == "narrow":
            goal_list = []
            goal_generator = lambda: [
            (round(random.uniform(0,5),1), round(random.uniform(5,10),1)),  # quadrant2
            (round(random.uniform(0,5),1), round(random.uniform(0,5),1))]   # quadrant3
            for i in range(env_num):
                goal_list.append(goal_generator()[(i+1)%2])
            env_list = [self.single_env(goal) for goal in goal_list]
            return env_list

        else:
            raise AttributeError("available arguments are: 'average' or 'random'")

if __name__ == "__main__":
    sampler = Environment_sampler()
    env_list = sampler.multi_env(env_num=4, goal_distribution="narrow")
    print(env_list)