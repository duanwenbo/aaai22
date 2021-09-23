"""
# time: 09/09/2021
# update: /
# author: Bobby
A simple test environment. Integrated functionalities include:
1. support inserting the additional state info (the agent goal)
2. support adding ostacles in the environment
"""
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import yaml

def load_env_config():
    with open("/home/gamma/wb_alchemy/sub_project/Configuration/environment.yml","r") as f:
        config = yaml.load(f.read())
        return config

class NavigationPro(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, goal, see_goal=False, obstacles="None") -> None:
        super(NavigationPro, self).__init__()
        """
        # A discrete environment with different level of obstacles generation. #
        self.goal: user need to assigned a specific goal when initializing this env within the boundary.
        self.see_goal: add this info into the state, this can significantly accelerate the learning.
        self.obstacles: defined different hard level of the environment by adding different obstacles
                        "None", "Easy", "Medium", "Hard". Details are availble in supplementary materials.
        """
        self.goal = goal  # should be a randomized tuple within the boundary
        self.see_goal = see_goal
        self.obstacles = obstacles
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([10,10])) if not see_goal else  Box(low=np.array([0,0,0,0]), high=np.array([10,10,10,10]))
        self.state =  np.array([9.9,9.9]) if not see_goal else np.array([9.9,9.9,self.goal[0], self.goal[1]])
        self.previous_state = 0.
        self.episode_length = 100
        self.action_dict = {"0":(0,0.1),"1":(0.1,0.1),"2":(0.1,0),"3":(0.1,-0.1),"4":(0,-0.1),"5":(-0.1,-0.1),"6":(-0.1,0),"7":(-0.1,0.1)}
        self.adsorption = False
        self.end = False
        self.distance = 10 # random initialize
        self.reward = 0
        self.env_config = load_env_config()[self.obstacles]

    def _detect_obstacles(self):
        """check the if the line was touched by the point"""
        def _distance(point, line_point1, line_point2):
            """calcuate the distance between a point and a line"""
            vec1 = line_point1 - point
            vec2 = line_point2 - point
            distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
            return distance

        def _acute_angle(point, line_point1, line_point2):
            """detetrmine if the point is whithin the boundary of the line through law of cosines"""
            base_line = np.linalg.norm(line_point1-line_point2)
            assert base_line > 0, "check the library useage"
            line1 = np.linalg.norm(point - line_point1)
            line2 = np.linalg.norm(point - line_point2)
            cos_angle_1 = (base_line**2 + line1**2 - line2**2)/(2*base_line*line1)
            cos_angle_2 = (base_line**2 + line2**2 - line1**2)/(2*base_line*line2)
            if cos_angle_1 * cos_angle_2 > 0:
                return True
            else:
                return False

        if self.obstacles != "None": # if user assigned some obstacles
            for line in self.env_config: 
                line_point1, line_point2 = np.array(line[0]), np.array(line[1])
                point = np.array(self.state[:2])
                distance = _distance(point, line_point1, line_point2)
                acute_angle = _acute_angle(point, line_point1, line_point2)
                if distance <= 0.02 and acute_angle:
                    self.adsorption = True
                    break
                else:
                    self.adsorption = False
    
    def _detect_stop(func):
        """There are two types of stop. one is on the end of episde, the other is when the point touched the goal or obstacles"""
        def wrapper(*args,**kwargs):
            self = args[0]
            self.episode_length -= 1
            if self.episode_length <=0:
                """if the episode is end"""
                self.end = True
            else:
                if self.adsorption:
                    """just stop moving and wait until the end of episode"""
                    self.state = self.previous_state
                else:
                    func(*args,**kwargs)
                    self._detect_obstacles()

                # func(*args,**kwargs)
                # self._detect_obstacles()
                # if self.adsorption:
                #     """if this step update is invalid, the point will rebond"""
                #     self.state = self.previous_state

            if self.distance <= 0.02:
                """if the point reached the boundary around the goal, let it stop and reset the punishment(self.reward)"""
                self.end = True
                self.reward = 0
            if self.state[0] <0 or self.state[0] > 10 or self.state[1] <0 or self.state[1] > 10:
                # self.end = True
                self.reward = -800
            return np.array(self.state), self.reward, self.end, self.distance
        return wrapper
    
    @ _detect_stop
    def step(self, action):
        x_increment, y_increment = self.action_dict[str(action)][0], self.action_dict[str(action)][1]
        self.previous_state = self.state
        if self.see_goal:
            self.state = (self.state[0]+ x_increment, self.state[1]+y_increment, self.goal[0], self.goal[1])
        else:
            self.state = (self.state[0]+ x_increment, self.state[1]+y_increment)
        self.distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)
        self.reward = (-self.distance) 

    def reset(self):
        self.state =  np.array([9.9,9.9]) if not self.see_goal else np.array([9.9,9.9,self.goal[0], self.goal[1]])
        self.episode_length = 100
        self.end = False
        self.adsorption = False
        return self.state

    def render(self):
        pass

if __name__ == "__main__":
    state = np.array([1,1])
    b = np.array([state[0]-1, state[1]-1])
    print(type(b))