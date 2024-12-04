#############################################################################################
# Start date: 11/11/2024                                                                    #
# Project name: Windy Gridworld                                                             #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################

#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random as random
import time

class const:
    @staticmethod
    def discount():
        return 1
    
    @staticmethod
    def max_row():
        return 7
    
    @staticmethod
    def max_col():
        return 10
    
# Miscellaneous function
def random_ele(array):
    random_index1 = np.random.randint(0, array.shape[0])
    return array[random_index1]

def compare_array(a, b):
    comparison = a == b
    equal_arrays = comparison.all()
    return equal_arrays

def check_off_map(state):
    if (state[0] < 0) or (state[0] > (const.max_row()-1))\
            or (state[1] < 0) or (state[1] > (const.max_col()-1)):
        return True
    else:
        return False
# Main 
class GridworldEnv:
    START = np.array([3, 0]) # (row,col) = (y, x)
    GOAL = np.array([3, 7])
    def __init__(self):
        self._actions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        self._actions_labels = {
            0: "down",
            1: "up",
            2: "right",
            3: "left"
        }
        self._windspeed = np.zeros((7,10))
        self._windspeed[:, 3:6] = 1
        self._windspeed[:, 6:8] = 2
        self._windspeed[:, 8] = 1 
        
    def step(self, state, action):
        terminal = False
        # state += action
        new_y = max(0, min(state[0] + action[0], const.max_row() - 1 ))
        new_x = max(0, min(state[1] + action[1], const.max_col() - 1 ))
        new_state = np.array([new_y, new_x])
        new_state[0] += -self._windspeed[new_y, new_x] # effect of the wind
        new_state[0] = max(0, new_state[0])
        if compare_array(new_state, self.GOAL):
            print("Agent reached the goal, terminate!")
            terminal = True
            reward = +0
        else:
            # print("Agent is finding the way...")
            reward = -1
        return (new_state, reward, terminal)
    
def trajectoryPath(traj):
    # Initialize gridworld
    world_map = np.zeros((const.max_row(), const.max_col()))
    for i,state in enumerate(traj):
        x = state[1]
        y = state[0]
        world_map[y, x] = i + 1
    print(world_map)
    print("\n")
        
def r_argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top, ties = q_values[i], [i]
        elif q_values[i] == top:
            ties.append(i)
    return np.random.choice(ties)



class TDAgent:
    def __init__(self, lambda_value, epsilon, alpha_t):
        self._lambda_value = lambda_value
        row, col = env._windspeed.shape
        self._q_values = np.zeros((row, col, len(env._actions)))
        self._epsilon = epsilon
        self._alpha_t = alpha_t
        self._value_visits = np.zeros((row, col))
        self._action_visits = np.zeros((row, col, len(env._actions)))
        self._eligibility_traces = np.zeros((row, col, len(env._actions)))
        self._default_visits = 100
        action = random.choice(env._actions)
    
    def take_index(self, target, list_action):
        index = 0
        for i, action in enumerate(list_action):
            if np.array_equal(action, target):
                index = i
                break
        return index
            
    def start(self, state):
        # action = np.array(np.random.choice(env._actions))
        action = random_ele(env._actions)
        act_index = self.take_index(action, env._actions)
        self._action_visits[state[0], state[1], act_index] += 1
        self._eligibility_traces[state[0], state[1], act_index] += 1
        self._last_state = state
        last_action = act_index
        self._last_state_action = \
            (self._last_state[0], self._last_state[1], last_action)
        return action
    
    def step(self, state, reward):
        self._alpha_t = 1/self._action_visits[self._last_state_action]
        self.epsilon = (self._default_visits / (self._default_visits + 
                np.sum(self._action_visits[self._last_state[0],\
                                          self._last_state[1]])))
        if np.random.random() < self.epsilon:
            action = random_ele(env._actions)
            act_index = self.take_index(action, env._actions)
        else:
            act_index = r_argmax(self._q_values[state[0], state[1]])
            action = env._actions[act_index]
        
        state_action = (state[0], state[1], act_index)

        delta = reward + const.discount()*self._q_values[state_action]\
            - self._q_values[self._last_state_action]

        temp = self._alpha_t * delta * self._eligibility_traces
        self._q_values += temp
        self._eligibility_traces *= const.discount()*self._lambda_value
        
        self._eligibility_traces[state_action] += 1
        self._action_visits[state_action] += 1
        self._value_visits[state[0], state[1]] += 1
        
        self._last_state = state
        self._last_state_action = (state[0], state[1], act_index)
        return action
    
    def end(self, reward):
        self._alpha_t = 1/ self._action_visits[self._last_state_action]
        delta = reward - self._q_values[self._last_state_action]
        self._q_values += self._alpha_t * delta * self._eligibility_traces       

def run_td_lambda(AgentClass, num_episodes):
    agent = AgentClass(lambda_value=0.01, epsilon=0.1, alpha_t=0.5)
    step = 0
    step_ep_list = []
    for episode in range(num_episodes):
        print("Episode:%d starts" %(episode))
        env = GridworldEnv()
        
        state = env.START.copy()
        trajectory = [state]
        # action = agent.start(state)
        if (episode == 0):
            action = agent.start(state)
        else:
            act_index = r_argmax(agent._q_values[state[0], state[1]])
            action = env._actions[act_index]

        while True:
            state, reward, terminal = env.step(state, action)
            step += 1
            step_ep_list.append(episode)
            trajectory.append(state)
            if not terminal:
                action = agent.step(state, reward)
            else:
                print(terminal)
                agent.end(reward)
                break

    start_time = time.time()
    plt.plot(step_ep_list)
    print("Time elapsed is (in Secs): ", time.time() - start_time)
    plt.title('WindyGridWorld_SARSA ', fontsize = 'large')
    plt.xlabel("Number of Steps taken")
    plt.ylabel("Number of Episodes")
    plt.show()
    # print(agent._q_values)
    print(trajectory)
    trajectoryPath(trajectory)
    # optimal_values = np.amax(agent._q_values, )
env = GridworldEnv()
run_td_lambda(TDAgent, 5)
