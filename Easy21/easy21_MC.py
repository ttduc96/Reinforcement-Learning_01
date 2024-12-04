#############################################################################################
# Start date: 18/10/2024                                                                    #
# Project name: Easy21                                                                      #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################


#IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class const:
    @staticmethod
    def gamma():
        return 1
    
    @staticmethod
    def win():
        return 1
    
    @staticmethod
    def lose():
        return -1
    
    @staticmethod
    def draw():
        return 0
    
    @staticmethod
    def player_continue():
        return 0

def draw_card(player_sum):
    card = np.random.randint(1, 11)
    color = np.random.randint(1, 4)
    if (color == 3):
        return player_sum - card
    else:
        return player_sum + card
    
def bust(player_sum):
    if (player_sum > 21) or (player_sum < 1):
        return 1
    else:
        return 0
    
def step(state, action): # take input (state, action) and put it into the environment to output the new state
    dealer_card, player_sum = state
    if (action == 1):
        player_sum = draw_card(player_sum)
        if bust(player_sum):
            return ((dealer_card, player_sum), const.lose(), True)
        else:
            return ((dealer_card, player_sum), const.player_continue(), False)
    elif (action == 0):
        dealer_sum = dealer_card
        while (dealer_sum < 17):
            dealer_sum = draw_card(dealer_sum)
            if bust(dealer_sum):
                return ((dealer_card, player_sum), const.win(), True)
        if (dealer_sum < player_sum):
            return ((dealer_card, player_sum), const.win(), True)
        elif (dealer_sum == player_sum):
            return ((dealer_card, player_sum), const.draw(), True)
        else:
            return ((dealer_card, player_sum), const.lose(), True)
            

def r_argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top, ties = q_values[i], [i]
        elif q_values[i] == top:
            ties.append(i)
    return np.random.choice(ties)

class MonteCarloAgent:
    def __init__(self):
        self.q_values = np.zeros((11, 22, 2))
        self.epsilon = 0
        self.alpha_t = 0
        self.action_visits = np.zeros((11, 22, 2))
        self.default_visits = 100
        self.state_action_history = None
        
    def start(self, state):
        action = np.random.randint(2)
        self.action_visits[state[0], state[1], action] += 1
        self.state_action_history = []
        self.state_action_history.append((state, action))
        self.last_state = state
        self.last_action = action
        return action
    
    def step(self, state): # take the current state and output the appropriate action based on greedy epsilon
        self.epsilon = (self.default_visits / (self.default_visits + 
                np.sum(self.action_visits[self.last_state[0],\
                                          self.last_state[1]])))
        if np.random.random() < self.epsilon:
            action = np.random.randint(2)
        else:
            action = r_argmax(self.q_values[state[0], state[1]])
        self.last_state = state
        self.state_action_history.append((state, action))
        return action
    
    def end(self, reward):
        self.last_state, self.last_action = self.state_action_history.pop(0)
        for state, action in self.state_action_history:
            last_state_action = (self.last_state[0],\
                                 self.last_state[1],\
                                 self.last_action)
            self.alpha_t = 1 / self.action_visits[last_state_action]
            self.q_values[last_state_action] += self.alpha_t * \
                    (reward - self.q_values[last_state_action])
            
            self.action_visits[state[0], state[1], action] += 1
            self.last_state = state
            self.last_action = action
            
        last_state_action = (self.last_state[0],\
                             self.last_state[1],\
                             self.last_action)
        self.alpha_t = 1 / self.action_visits[last_state_action]
        self.q_values[last_state_action] += self.alpha_t * \
                (reward - self.q_values[last_state_action])
                
monteCarloAgent = MonteCarloAgent()
for episode in range(100000):
    state = (np.random.randint(1, 11), np.random.randint(1, 11))
    action = monteCarloAgent.start(state)
    terminal = False
    while True:
        state, reward, terminal = step(state, action)
        if not terminal:
            action = monteCarloAgent.step(state)
        else:
            monteCarloAgent.end(reward)
            break
        
optimal_values = np.amax(monteCarloAgent.q_values, axis=2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = []
y = []
for i in range(1, 11):
    for j in range(1, 22):
        x.append(i)
        y.append(j)
z = optimal_values[1:, 1:].flatten()
ax.set_xlabel('dealer card')
ax.set_ylabel('player sum')
ax.set_zlabel('value')
ax.set_title('V*(s) for Monte Carlo')
surface = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm)
plt.xticks(np.arange(1, 11))
plt.yticks(np.arange(1, 22, 3))
plt.show()