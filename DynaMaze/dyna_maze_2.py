#############################################################################################
# Start date: 15/11/2024                                                                    #
# Project name: Dyna Maze                                                                   #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################

import gym
import gym.spaces
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random as random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import defaultdict
array = [1, 2, 3]
class DynaMaze(gym.Env):
    #Initialize
    # SIZE = (9, 6) #(col, row) (x, y)
    # START = (0, 2)
    # GOAL = (8, 0)
    
    SIZE = (6, 9) #(col, row) (x, y)
    START = (5, 0)
    GOAL = (0, 8)
    
    wall = np.empty((0,2))
    print("my size is:", SIZE)
    map = np.zeros([6, 9])
    map[4, 2] = map[5, 2] = 1
    map[1, 2] = map[2, 2] = 1
    map[4, 5] = 1
    map[0, 7] = map[1, 7] = map[2, 7] = 1
    print(map)
    for row in range(SIZE[0]):
        for col in range(SIZE[1]):
            if map[row, col] == 1:
                wall = np.vstack((wall, np.array([row, col])))
    ACTIONS = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    
    observation_space = gym.spaces.MultiDiscrete(SIZE)
    reward_range = (-1, 1)
    
    def __init__(self, stop=False):
        self.stop = stop
        
        self.actions = self.ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        
        self.position = None
        self.arrow = None
        
        self.ax = None

    def step(self, action):
        assert self.action_space.contains(action)
        
        #Calculate move and position
        delta = self.actions[action]
        # print(delta)
        position = self.position + np.array(delta)
        # print("position before check", position)

        if np.any(np.all(position == self.wall, axis = 1)):
            # print("hit the wall at:", position)
            # print("The wall is:", self.wall)
            position = self.position #Stay at the previous position if hits the wall
            # print("still stay at:", position)
        
        
        #Store position and calculate arrow
        position = np.clip(position, 0, self.observation_space.nvec - 1)
        self.arrow = position - self.position
        self.position = position
        # print("position after check",position)
        # print(isinstance(position, np.ndarray))
        
        #Check terminal state
        terminated = np.array_equal(self.position, self.GOAL)
        # reward = +1 if terminated else -1
        if terminated:
            reward = +1
            # print("Reached the GOAL!")
        else:
            reward = -1
        
        # print("Position is:", position)
        assert self.observation_space.contains(position)
        return position, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #Choose the location
        self.position = np.array(self.START)
        # print("starting pos is:", self.position)
        self.arrow = np.array((0, 0))
        
        self.ax = None
        
        return self.position
    
    def render(self, mode='human'):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()
            
            #Notation for graph
            self.ax.annotate("G", (self.GOAL[1], self.GOAL[0]), size = 25, color='gray', ha='center', va='center')
            self.ax.annotate("S", (self.START[1], self.START[0]), size = 25, color='gray', ha='center', va='center')
            
            #Background color for wall
            self.ax.imshow(self.map, aspect='equal', origin='upper', cmap='Blues')
            
            #Thin grid line
            self.ax.set_xticks(np.arange(-0.5, self.SIZE[1]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.SIZE[0]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)
            
        #Arrow pointing from the previous to the current position
        if np.all(self.arrow == 0):
            patch = mpatches.Circle([self.position[1], self.position[0]], radius=0.05, color='black', zorder=1)
        else:
            y_base, x_base = self.position - self.arrow
            dy, dx = self.arrow
            patch = mpatches.FancyArrow(x_base, y_base,\
                        dx, dy, color='black',\
                        zorder=2, fill=True, width=0.05,\
                        head_width=0.25,\
                        length_includes_head=True)
            
        self.ax.add_patch(patch)
        
gym.envs.registration.register(
    id='DynaMaze',
    entry_point=lambda stop:DynaMaze(stop),
    kwargs={'stop':False},
    max_episode_steps = 5000,
)

def dyna(env, num_episodes, epsilon_0, alpha, eval_epochs, gamma):
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete
   
    #Number of available action and maximum state ravel index
    n_action = env.action_space.n
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1
    state_memory = []
    action_memory = {}
    
    q = np.zeros([n_state_ridx, n_action], dtype=np.float64)
    policy = np.ones([n_state_ridx, n_action], dtype=np.float64) / n_action
    model = {}
    
    history = [0] * num_episodes
    
    for episode in range(num_episodes):
        #Reset
        step_count = 0
        state = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.random.choice(n_action, p=policy[state_ridx])

        done = False
        print("I am at episode:", episode)
        while not done:
            #First store the state and action to memory
            state_memory.append(state_ridx)
            if state_ridx not in action_memory:
                action_memory[state_ridx] = []
            action_memory[state_ridx] += [action]
            
            if state_ridx not in model.keys():
                model[state_ridx] = {}
            
            #Run the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(next_state)
            step_count += 1
            done = terminated or truncated
            next_state_ridx = np.ravel_multi_index(next_state, env.observation_space.nvec)
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])
            
            #Update Q value
            q[state_ridx, action] += alpha*(reward + gamma * np.max(q[next_state_ridx, :]) - q[state_ridx, action])
            
            #Epsilon greedy policy
            epsilon = epsilon_0/(episode + 1)
            policy[state_ridx, :] = epsilon/n_action
            policy[state_ridx, np.argmax(q[state_ridx, :])] = 1 - epsilon + epsilon/n_action
            assert np.allclose(np.sum(policy, axis=1), 1)
            
            #Update model
            model[state_ridx][action] = (reward, next_state_ridx)
            
            #Planning with the model
            for i in range(eval_epochs):
                state_plan = np.random.choice(state_memory)
                action_plan = np.random.choice(action_memory[state_plan])
                reward_plan, next_state_plan = model[state_plan][action_plan]
                q[state_plan, action_plan] += alpha*(reward_plan + gamma * np.max(q[next_state_plan, :]) - q[state_plan, action_plan])
           
            #Prepare the next q
            state_ridx = next_state_ridx
            action = next_action
            history[episode] += 1
            
    return q, policy, history

def run_episode(env, policy=None, render=True):
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete
    
    state = env.reset()
    if render:
        env.render()
    done = False
    rewards = []
    print(env.map)
    print(env.wall)
    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.argmax(policy[state_ridx])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards += [reward]
        
        if render:
            env.render()
    
    if render:
        plt.show()
        
    return rewards

env = gym.make('DynaMaze')
q, policy, history = dyna(env, 200, epsilon_0=0.1, alpha=0.1, eval_epochs=20, gamma=0.95)
print(history)
plt.figure()
# plt.xlabel('Time steps'); plt.xlim(0, 8000)
# plt.ylabel('Episodes'); plt.ylim(0, 300)
plt.ylabel('Time steps'); plt.ylim(0, 6000)
plt.xlabel('Episodes'); plt.xlim(0, 200)
timesteps = np.cumsum([0] + history)
# plt.plot(timesteps, np.arange(len(timesteps)),  color='red')
plt.plot(np.arange(len(timesteps)), timesteps,  color='red')
plt.show()
rewards = run_episode(env, policy, render=True)
# plot_results(env, q, policy)



