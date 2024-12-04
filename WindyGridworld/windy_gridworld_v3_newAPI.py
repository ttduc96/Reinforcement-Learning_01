#############################################################################################
# Start date: 15/11/2024                                                                    #
# Project name: Windy Gridworld v2                                                          #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################

#IMPORTS
import gym.spaces
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random as random
import time
import gym
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class GridworldEnv(gym.Env):
    SIZE = (10, 7) # (col, row) (x, y)
    START = (0, 0)
    GOAL = (7, 3)
    
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    PAWN_ACTIONS = [(0, +1), (+1, 0), (0, -1), (-1, 0)]
    
    observation_space = gym.spaces.MultiDiscrete(SIZE) # CHECK
    reward_range = (-1, -1)
    
    def __init__(self, king=False, stop=False, stochastic=False):
        self.stochastic = stochastic
        self.king = king
        self.stop = stop
        
        self.actions = self.PAWN_ACTIONS[:]
        if self.king:
            self.actions += self.KINGS_ACTION
        if self.stop:
            self.actions += self.STOP_ACTION
        self.action_space = gym.spaces.Discrete(len(self.actions)) # CHECK, khai bao voi gym tao ra 1 list action tu 0,1,2...

        self.position = None
        self.arrow = None
        
        self.ax = None
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        #Calculate move and position
        delta = self.actions[action]
        position = self.position + np.array(delta)
        
        #Add wind
        wind = self.WIND[self.position[0]]
        if self.stochastic and wind > 0:
            wind += np.random.choice([-1, 0, 1])
        position[1] += wind
        
        #Store position and calculate arrow
        position = np.clip(position, 0, self.observation_space.nvec - 1)
        self.arrow = position - self.position
        self.position = position
        
        #Check terminal state
        terminated = np.array_equal(self.position, self.GOAL)
        reward = 0 if terminated else -1
       
        
        assert self.observation_space.contains(position)
        return position, reward, terminated, False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #Choose the location
        self.position = np.array(self.START)
        self.arrow = np.array((0, 0))

        self.ax = None
        
        return self.position
    
    def render(self, mode='human'):
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.gca()
            
            #Background color for wind strength
            wind = np.vstack([self.WIND] * self.SIZE[1]) #Create a wind map 7x10
            self.ax.imshow(wind, aspect='equal', origin='lower', cmap='Blues')
            
            self.ax.annotate("G", self.GOAL, size = 25, color='gray', ha='center', va='center')
            self.ax.annotate("S", self.START, size = 25, color='gray', ha='center', va='center')
            
            #Tick marks showing wind strength
            self.ax.set_xticks(np.arange(len(self.WIND)))
            self.ax.set_xticklabels(self.WIND)
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])
            
            #Thin grid line at minor tick marks
            self.ax.set_xticks(np.arange(-0.5, self.SIZE[0]), minor=True)
            self.ax.set_yticks(np.arange(-0.5, self.SIZE[1]), minor=True)
            self.ax.grid(which='minor', color='black', linewidth=0.20)
            self.ax.tick_params(which='both', length=0)
            self.ax.set_frame_on(True)
            
        #Arrow pointing from the previous to the current position
        if np.all(self.arrow == 0):
            patch = mpatches.Circle(self.position, radius=0.05, color='black', zorder=1)
        else:
            x_base, y_base = self.position - self.arrow
            dx, dy = self.arrow
            patch = mpatches.FancyArrow(x_base, y_base,\
                                        dx, dy, color='black',\
                                        zorder=2, fill=True, width=0.05,\
                                        head_width=0.25,\
                                        length_includes_head=True) # (self.position - self.arrow) is to determined the base of the arrow
            
        self.ax.add_patch(patch)
            
gym.envs.registration.register(
    id='WindyGridworld-v2',
    entry_point=lambda king, stop, stochastic: GridworldEnv(king, stop, stochastic),
    kwargs={'king': False, 'stop': False, 'stochastic': False},
    max_episode_steps = 5000,
)

def sarsa(env, num_episodes, epsilon_0, alpha):
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete
    
    #Number of available action and maximum state ravel index
    n_action = env.action_space.n
    n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1
    
    q = np.zeros([n_state_ridx, n_action], dtype=np.float64)
    policy = np.ones([n_state_ridx, n_action], dtype=np.float64) / n_action
    history = [0] * num_episodes
    for episode in range(num_episodes):
        #Reset the environment
        state = env.reset()
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec) #check
        action = np.random.choice(n_action, p=policy[state_ridx])        
        done = False
        print("I am at episode:", episode)
        while not done:
            #Run the environment
            # next_state, reward, done, info = env.step(action)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state_ridx = np.ravel_multi_index(next_state, env.observation_space.nvec)
            next_action = np.random.choice(n_action, p=policy[next_state_ridx])
            
            #Update Q value
            q[state_ridx, action] += alpha*(reward + q[next_state_ridx, next_action] - q[state_ridx, action])
 
            #Extract epsilon-greedy policy from the updated q
            epsilon = epsilon_0/(episode + 1)
            policy[state_ridx, :] = epsilon/n_action
            policy[state_ridx, np.argmax(q[state_ridx])] = 1 - epsilon + epsilon/n_action
            assert np.allclose(np.sum(policy, axis=1), 1)
            
            #Prepare the next q
            state_ridx = next_state_ridx
            action = next_action
            history[episode] += 1
            
    return q, policy, history

def run_episode(env, policy=None, render=True):
    """ Follow policy through an environment's episode and return an array of collected rewards """
    assert type(env.action_space) == gym.spaces.Discrete
    assert type(env.observation_space) == gym.spaces.MultiDiscrete

    state = env.reset()
    if render:
        env.render()

    done = False
    rewards = []
    while not done:
        state_ridx = np.ravel_multi_index(state, env.observation_space.nvec)
        action = np.argmax(policy[state_ridx])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards += [reward]

        if render:
            env.render()

    if render:
        import matplotlib.pyplot as plt
        plt.show()

    return rewards

matplotlib.rcParams['figure.figsize'] = [10, 10]


def plot_results(env, q, policy):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Optimal Value Function and Policy")
    
    q = np.copy(q)
    unvisited = np.where(q == 0)
    q[unvisited] = -np.inf
    v = np.max(q, axis=1).reshape(env.observation_space.nvec)
    ax.imshow(v.T, origin='lower')

    a_stars = np.argmax(policy, axis=1)
    arrows = np.array([env.actions[a] for a in a_stars])
    arrows[unvisited[0], :] = 0
    arrows = arrows.reshape([*env.observation_space.nvec, 2])
    xr = np.arange(env.observation_space.nvec[0])
    yr = np.arange(env.observation_space.nvec[1])
    ax.quiver(xr, yr, arrows[:, :, 0].T, arrows[:, :, 1].T, pivot='mid')
    
env = gym.make('WindyGridworld-v2')
q, policy, history = sarsa(env, 120, epsilon_0=0.5, alpha=0.5)
plt.figure()
plt.xlabel("Time steps"); plt.xlim(0, 8_000)
plt.ylabel("Episodes"); plt.ylim(0, 170)
timesteps = np.cumsum([0] + history)
plt.plot(timesteps, np.arange(len(timesteps)), color='red')
plt.show()
plot_results(env, q, policy)
rewards = run_episode(env, policy, render=True)
# print(f"Episode length = {len(rewards)}")